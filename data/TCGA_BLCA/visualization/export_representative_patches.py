import pickle
import os
import glob

import h5py
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

class CN_patches():
    def __init__(self,CN_file,color_map):
        with h5py.File(CN_file,'r') as f:
            self.imgs = f['image_array'][:]
            self.coords = f['coords'][:]
        with open(color_map,'rb') as f:
            self.representative_patches_pos = pickle.load(f)[2]
        self.patch_list = []

    def get_representative_patch(self):
        ind_map = {tuple(item):idx for idx,item in enumerate(map(tuple,self.coords))}
        positions = [ind_map.get(tuple(item),None) for item in map(tuple,self.representative_patches_pos)]
        for i in positions:
            self.patch_list.append(self.imgs[i])
        return self.patch_list
    
def get_CNpatches(slide_id):
    CN_file = os.path.join('data/TCGA_bladder_threshold_80/CN_patches/size512/10X',slide_id+'.h5')
    color_map = os.path.join('data/TCGA_bladder_threshold_80/visualization/attention_map/best_score',slide_id+'.pkl')
    save_folder = os.path.join('data/TCGA_bladder_threshold_80/visualization/representative_CN_patches/',slide_id)
    os.makedirs(save_folder,exist_ok=True)
    patches = CN_patches(CN_file,color_map).get_representative_patch()
    for i,patch in enumerate(patches):
        save_path = save_folder + '/' + f'{slide_id}_{i}.png'
        plt.imsave(save_path,patch)

def main():
    attention_map_folder = Path('data/TCGA_bladder_threshold_80/visualization/attention_map/best_score')
    attention_maps = glob.glob(os.path.join(attention_map_folder,'*.pkl'))
    slide_ids = [Path(i).stem for i in attention_maps]
    for slide in slide_ids:
        get_CNpatches(slide)
        print(f'{slide} exported')

if __name__ == '__main__':
    main()
