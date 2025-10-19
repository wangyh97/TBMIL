import os
import glob
from pathlib import Path
import multiprocessing
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import tifffile as tif

def path_to_img_array(uuid, scale, type: str = 'T',save=True):
    img_array_ls = []
    unread_array_uuids = []
    assert type in ['T','nonT'], 'invalid tissue type'
    path = f'/GPUFS/sysu_jhluo_1/wangyh/data/patches_annotated/CN_patches/*/{uuid}/{scale}X/'
    try:
        tifs = glob.glob(path + f'{type}*.tiff')
        if len(tifs) ==0:
            print(f'{uuid}:{scale}X -- no CNed patches')
        else:
            dir_path = Path(tifs[0]).parent
            for t in tifs:

                img = tif.imread(t)
                tif_coord = Path(t).stem.split('_')[1] + '_' + Path(t).stem.split('_')[2]
                img_array_ls.append((tif_coord,img))
                img_array = np.array(img_array_ls)
            if save:
                saving_path = str(dir_path) + f'_array.npy'
                np.save(saving_path, img_array)
            return img_array
    except Exception as e:
        print(f'wrong in func<path_to_img_array> when handle uuid {uuid}, log: {e}')
        unread_array_uuids.append(uuid)
        np.save('unread_array_uuids.npy',unread_array_uuids)
        return None

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='img2array')
    
    parser.add_argument('-n','--num_workers', default=10, type=int)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=1)
    
    args = parser.parse_args()
    
    
    full = pd.read_csv('../config/patch_info.csv')
    end = min(args.end,full.shape[0])

    uuids = full['dir_uuid'][args.start:end]

    
    pool = multiprocessing.Pool(processes=args.num_workers) 
    
    for uuid in tqdm(uuids):
        for scale in [10,20]:
            _ = pool.apply_async(path_to_img_array, (uuid, scale))
    pool.close()
    pool.join()