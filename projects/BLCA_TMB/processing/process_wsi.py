import time
import os
import glob
import gc
import argparse
from multiprocessing import Pool

from pathlib import Path
import openslide
import h5py
from tqdm import tqdm
from memory_profiler import profile

import matplotlib.pyplot as plt
import pandas as pd

import extract_patches_nonT as ep


def slide_parser(uuid):
    folder_path = f'data/slides/TCGA_bladder/TCGA_bladder/{uuid}'
    try:
        slide_path = glob.glob(folder_path + '/*.svs')
        xml_path = glob.glob(folder_path + '/*.xml')
        patch_path = f'data/raw_patches/TCGA_bladder/{uuid}'
        Path(patch_path).mkdir(parents=True,exist_ok=True)
        return slide_path[0], xml_path[0],patch_path
    except Exception as e:
        print(f'cannot handle {uuid},reason:{e}')
        return None

@profile    
def extract_thumbnails(slide,saving_path,xml_path,rule):
    tick = time.time()
    
    thumbnail = slide.get_thumbnail((512,512))
#     mask_coords,classes = ep.AnnotationParser(xml_path)
    annos = ep.Annotation(slide,xml_path,rule=rule)
    tumor_slide,non_tumor_slide = ep.get_mask_slide(annos)
    T_marked_thumb = tumor_slide.get_thumbnail((512,512)) #masked slides are images range in [0,1],use pixel_255 for better visualization
    non_T_marked_thumb = non_tumor_slide.get_thumbnail((512,512))

    with h5py.File(saving_path,'w') as f:
        f.create_dataset('thumbnail',data = thumbnail)
        f.create_dataset('tumor_mask_thumb',data=T_marked_thumb)
        f.create_dataset('nonTumor_mask_thumb',data = non_T_marked_thumb)
    print('h5py saved')
        
    del annos, tumor_slide, non_tumor_slide, thumbnail, T_marked_thumb, non_T_marked_thumb
    gc.collect()
    
    print(f'{uuid}---->consuming{time.time()-tick}s')
    
    
def get_wsi_with_mark(uuid,thumbnail_path,cmap = 'gray'):
    
    
    plt.rcParams['image.cmap'] = cmap

    saving_path = os.path.join(thumbnail_path,uuid + '.h5')
    if not os.path.exists(saving_path):

        svs_path, xml_path,_ = slide_parser(uuid)
        rule =  {"tumor":{"excludes":["artificial","stroma","necrosis"]},
        'stroma':{"excludes":['artificial','necrosis']}}
        if svs_path:
            with openslide.OpenSlide(svs_path) as slide:
                try:
                    extract_thumbnails(slide,saving_path,xml_path,rule)    
                except Exception as e:
                    print(f'Anno & get_thumb error! {uuid} error occured:{e}')
    else:
        print(f'{uuid} has already been processed')
            
    
        
def process_wsi(uuid):
    try:
        get_wsi_with_mark(uuid, thumbnail_path, cmap='gray')
        print(f'{uuid} done')
    except Exception as e:
        print(f'{uuid} error as {e}')

# generate unit test case,uuids from TCGA bladder cancer

if __name__ == '__main__':
    uuids = ['474e93f2-2ee8-478d-9a87-ab561286535f', 'f9cee804-1f30-4a97-b5ae-55cf24ad220e', 'f4ca3ddd-dc53-4ab0-b55b-942603b64e57', '7a0697d9-18db-4e52-b243-be06879a9944']
    print(uuids)
    
    N = len(uuids)
    print(f'{N} slides in total')
    
    thumbnail_path = 'data/TCGA_bladder/thumbnails'
    Path(thumbnail_path).mkdir(parents=True,exist_ok=True)

    for uuid in tqdm(uuids):
        try:
            get_wsi_with_mark(uuid,thumbnail_path,cmap='gray')
            print(f'{uuid} done')
        except Exception as e:
            print(f"{uuid} error as {e}")


    print('all done')

