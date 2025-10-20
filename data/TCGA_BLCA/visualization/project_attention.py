import os
import glob
import pickle
from itertools import product
from PIL import Image

import h5py
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator

def get_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide

def get_DZG(slide,tile_size=512,overlap=False,limit_bounds=False,slide_tile = False):
    dzg = DeepZoomGenerator(slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    return dzg

def get_thumbnail(slide_id):
    with open('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/misc_files/uuid_SlideID_TMB.pkl','rb') as f:
        uuid_slideId_TMB = pickle.load(f)
        slideId_uuid_dict = dict(zip(uuid_slideId_TMB['slide_id'],uuid_slideId_TMB['uuid']))

        uuid = slideId_uuid_dict[slide_id]
    thumbnail_path = os.path.join('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/thumbnails',uuid + '.h5')
    with h5py.File(thumbnail_path,'r') as f:
        thumbnail = f['thumbnail'][:]
    return thumbnail

def get_thumbnail_from_scratch(slide,cols,rows,min_edge = 512):
    scale_factor = min_edge / min(cols, rows)
    x_new = cols * scale_factor
    y_new = rows * scale_factor    
    return slide.get_thumbnail(size = (x_new,y_new)),scale_factor

def get_col_rows(dzg,level):
    cols,rows = dzg.level_tiles[level]
    return cols,rows

def get_level(dzg,scale):
    level_count = dzg.level_count
    levels=[level_count-3,level_count-2]  # 10,20
    level_dict = {
        '10X': levels[0],
        '20X': levels[1]
    }
    return level_dict[scale]

def get_color_map(color_map_path):
    with open(color_map_path,'rb') as f:
        color_map,pos_arr,_ = pickle.load(f)
    return color_map,pos_arr

def project_color_map(slide,color_map,pos_arr,dzg,level):
    ALPHA = 0.3

    dzg = get_DZG(slide)
    cols,rows = get_col_rows(dzg,level)
    thumbnail,scale_factor = get_thumbnail_from_scratch(slide,cols,rows)
    thumbnail = np.array(thumbnail)
    resized_color_map = np.zeros((thumbnail.shape[0],thumbnail.shape[1],3),dtype = np.float)
    for col,row in product(range(cols),range(rows)):
        start_x = int(scale_factor * col)
        start_y = int(scale_factor * row)

        end_x = int(start_x + scale_factor)
        end_y = int(start_y + scale_factor)
        try:
            resized_color_map[start_y:end_y, start_x:end_x, :] = color_map[row,col,:]*255
        except:
            pass
    mapped = thumbnail * (1-ALPHA) + resized_color_map*ALPHA
    image_processed = np.clip(np.round(mapped), 0, 255).astype(np.uint8)
    return image_processed
            
def main():
    OVERLAP =0
    LIMIT = False
    scales = ['10X','20X']
    TILE_SIZE = 512
    
    attention_map_folder = Path('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder_threshold_80/visualization/attention_map/best_score')
    attention_maps = glob.glob(os.path.join(attention_map_folder,'*.pkl'))
    slide_ids = [Path(i).stem for i in attention_maps]
    
    slide_folder = '/GPUFS/sysu_jhluo_1/wangyh/data/slides/TCGA_bladder/TCGA_bladder/'
    slide_paths = [glob.glob(os.path.join(slide_folder,f'*/{slide_id}.svs'))[0] for slide_id in slide_ids]

    save_folder = '/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder_threshold_80/visualization/projected_attention_maps'

    for slide_path in slide_paths:
        slide_id = Path(slide_path).stem
        slide = get_slide(slide_path)
        dzg = get_DZG(slide)
        level = get_level(dzg,'10X')
        color_map,pos_arr = get_color_map(os.path.join(attention_map_folder,slide_id+'.pkl'))

        mapped_slides = project_color_map(slide,color_map,pos_arr,dzg,level)
        save_path = os.path.join(save_folder,slide_id+'.png')

        plt.imsave(save_path,mapped_slides)
        print(f'{slide_id} projected')

if __name__ == '__main__':
    main()
    

