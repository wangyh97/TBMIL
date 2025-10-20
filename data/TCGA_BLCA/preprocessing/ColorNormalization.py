#!/usr/bin/env python
# coding: utf-8

import os
import logging
import gc
import argparse
from PIL import Image
import pickle

import pandas as pd 
import numpy as np
from pathlib import Path
import h5py
import cupy as cp
import tifffile as tif
from tqdm import tqdm

# import staintools

def pixel_255(image):
    assert type(image) is np.ndarray
    image[image==0] = 255
    return image

def gen_normal_size(I_list,target_shape=(512,512,3)):
    for i in I_list:
        try:
            tiff = tif.imread(i)
            if tiff.shape == target_shape:
                yield i
        except Exception as e:
            print(e)
            pass

class HENormalizer:
    def fit(self, target):
        pass

    def normalize(self, I, **kwargs):
        raise Exception('Abstract method')

"""
Inspired by torchstain :
Source code adapted from: https://github.com/schaugf/HEnorm_python;
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class Normalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = cp.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = cp.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -cp.log((I.astype(cp.float32)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~cp.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = cp.arctan2(That[:,1],That[:,0])

        minPhi = cp.percentile(phi, alpha)
        maxPhi = cp.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(cp.array([(cp.cos(minPhi), cp.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(cp.array([(cp.cos(maxPhi), cp.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = cp.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = cp.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = cp.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = cp.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)
        assert len(ODhat)!=0,'all pixels are transparent'
        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = cp.array([cp.percentile(C[0,:], 99), cp.percentile(C[1,:],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        I = cp.asarray(I)
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15):
        ''' Normalize staining appearence of H&E stained images
        Example use:
            see test.py
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity
        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image
        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
       # I = cp.asarray(I)
        batch,h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = cp.divide(maxC, self.maxCRef)
        C2 = cp.divide(C, maxC[:, cp.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = cp.multiply(Io, cp.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = cp.reshape(Inorm.T, (batch,h, w, c)).astype(cp.uint8)

        return Inorm

def init_logger(description):
    logger = logging.getLogger('CN_logger')
    handler = logging.FileHandler(filename=f'CN_{description}.log')
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
 
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-i','--batch',type=int,help='batch index to be CNed')
parser.add_argument('-n','--size',type=int,help='size of the batch')
parser.add_argument('-r','--remove',action='store_true',help='delete raw patches after CN')
parser.add_argument('--description',default=None,type=str)
parser.add_argument('--gpu',default=0, type=int,help='gpu index')

args = parser.parse_args()

if args.description:
    description = args.description
else:
    description = f'batch{args.batch}_size{args.size}'

logger = init_logger(description)
logger.info(f'logger init: {description}')

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

cases = list(Path('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/raw_patches/size512').rglob('*.h5'))
SIZE = 512

cases_select = cases[args.size*(args.batch-1):args.size*args.batch]

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

template_file = np.load('/GPUFS/sysu_jhluo_1/wangyh/data/CN_patches/CN_template.npy')
normalizer = Normalizer()
normalizer.fit(template_file)

unnormed_cases = {
        'no_tiles':[],
        'invalid_file':[]
    }

scales = ['10X','20X']

for case in tqdm(cases_select):
    # 创建对应的CN文件路径
    cn_tiles_path = str(case).replace('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/raw_patches/','/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/CN_patches/') #TODO
    cn_folder_path = Path(cn_tiles_path).parent
    cn_folder_path.mkdir(parents=True, exist_ok=True)
    
    # 对应h5没有被创建，则执行CN
    if Path(cn_tiles_path).exists():
        logger.info(f"already processed case :{case.stem},scale:{str(case.parent).split('/')[-1]}")
    else:
        try:
            with h5py.File(case, 'r') as f:
                tiles = f['image_array'][:]
                coords = f['coords'][:]

                normed_images = []
                normed_coords = []

                print(len(tiles))

                if len(tiles):
                    for tile,coord in zip(tiles,coords):
                        try:
                            tile_img = pixel_255(tile)
                            if tile_img.shape[0] != SIZE or tile_img.shape[1] != SIZE:
                                tile_img = Image.fromarray(np.uint8(tile_img)).resize((SIZE,SIZE),Image.ANTIALIAS)
                                tile_img = np.asarray(tile_img)

                            tile_img = tile_img.reshape(1,512,512,3)
                            imgs = cp.asarray(tile_img,dtype=cp.float64)
                            norm_imgs= cp.asnumpy(normalizer.normalize(I=imgs))
                            norm_imgs = norm_imgs.reshape(512,512,3)
                            normed_images.append(norm_imgs)
                            normed_coords.append(coord)

                        except Exception as e:
                            logger.info(f"unprocessed tile in case :{case.stem},scale:{str(case.parent).split('/')[-1]}, coords:{coord}")

                        imgs = None
                        norm_imgs = None
                        mempool.free_all_blocks() 
                        pinned_mempool.free_all_blocks()
                        gc.collect()
                    with h5py.File(cn_tiles_path, 'w') as f:
                        f.create_dataset('image_array', data=np.array(normed_images))
                        f.create_dataset('coords', data=np.array(coords))

                    logger.info(f"processed case:{case},scale:{str(case.parent).split('/')[-1]}")
                else:
                    logger.info(f'no patches in case:{case}, may be removed') 
                    unnormed_cases['no_tiles'].append(case)
        except Exception as e:
            logger.info(f"unprocessed case:{case},scale:{str(case.parent).split('/')[-1]},error as {e}")
            unnormed_cases['invalid_file'].append(case)
            continue

with open(f'/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/CN_patches/unnormed_{args.batch}.pkl','wb') as f:
    pickle.dump(unnormed_cases,f)


