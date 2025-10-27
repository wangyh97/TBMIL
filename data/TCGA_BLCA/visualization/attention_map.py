import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import random
import logging
import pickle
import pandas as pd
import numpy as np
import h5py
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
from pathlib import Path

with open('data/TCGA_bladder/misc_files/uuid_SlideID_TMB.pkl','rb') as f:
    uuid_slideId_TMB = pickle.load(f)
    slideid_label_dict = dict(zip(uuid_slideId_TMB['slide_id'],uuid_slideId_TMB['TMB']))
    uuid_slideId_dict = dict(zip(uuid_slideId_TMB['uuid'],uuid_slideId_TMB['slide_id']))
    uuid_label_dict = dict(zip(uuid_slideId_TMB['uuid'],uuid_slideId_TMB['TMB']))

def set_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
    print('seed set')

class BagDataset(): # one bag with instances
    def __init__(self, bag_info, transform=None):
        self.features = bag_info['features'] # features
        self.coords = bag_info['coords'] # coords
        self.transform = transform
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        img = self.features[idx] # should be array
        img_pos = self.coords[idx] # should be array,row_col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, bag_info):
    transformed_dataset = BagDataset(bag_info=bag_info,
#                                     transform=Compose([
#                                         ToTensor()
#                                     ])
    )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def load_model(args):
    weight = torch.load(args.weight_path)

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    milnet = torch.nn.DataParallel(milnet)
    milnet = milnet.cuda()
    milnet.load_state_dict(weight, strict=True)
    return milnet

def test(args, bags, milnet):
    milnet.eval()
    num_bags = len(bags.keys())
    Tensor = torch.FloatTensor
    colors = [np.array((238,44,44)),np.array((255,106,106))]
    threshold_path = Path(args.weight_path).with_suffix('.pkl')
    with open(threshold_path,'rb') as f:
        threshold = pickle.load(f)
    for slide_id in bags.keys():
        feats_list = []
        pos_list = []
        classes_list = []
        test_predictions = []
        bag_info = bags[slide_id]
        dataloader, bag_size = bag_dataset(args, bag_info)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                feats = batch['input'].float().cuda()
                patch_pos = batch['position']

                feats, classes = milnet.i_classifier(feats)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
                pos_arr = np.vstack(pos_list)
                feats_arr = np.vstack(feats_list)
                classes_arr = np.vstack(classes_list)
                bag_feats = torch.from_numpy(feats_arr).cuda()
                ins_classes = torch.from_numpy(classes_arr).cuda()
                bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()

            if len(bag_prediction.shape)==0 or len(bag_prediction.shape)==1:
                bag_prediction = np.atleast_1d(bag_prediction)
            benign = True

            attentions = A[:, 1].cpu().numpy()
            max_indices = np.argsort(attentions)[min(10,len(attentions)):]
            highly_attented_pos = [pos_arr[i] for i in max_indices]
            colored_tiles = np.matmul(attentions[:, None], colors[1][None, :])
            if bag_prediction[1] >= threshold[1]: 
#                 num_pos_classes += 1
                print(slide_id + ' is detected as: ' + args.class_name[1])
            else:
                print(slide_id + ' is detected as: ' + args.class_name[0])      

                
#             colored_tiles = (colored_tiles / num_pos_classes) # 给多分类用的
            colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))
            
            color_map = np.zeros((np.amax(pos_arr, 0)[0]+1, np.amax(pos_arr, 0)[1]+1, 3)) # 找到最大的patch的横纵坐标，创建一个空的colormap
            
            for k, pos in enumerate(pos_arr):
                color_map[pos[0], pos[1]] = colored_tiles[k]
            slide_name = slide_id
#             color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
#             io.imsave(os.path.join(args.map_path, Path(args.weight_path).stem,slide_name+'.png'), img_as_ubyte(color_map))
            io.imsave(os.path.join(args.map_path, Path(args.weight_path).stem,slide_name+'.png'), img_as_ubyte(color_map))
            with open(os.path.join(args.map_path, Path(args.weight_path).stem, slide_name+'.pkl'),'wb') as f:
                pickle.dump((color_map,pos_arr,highly_attented_pos),f)
#             if args.export_scores:
#                 df_scores = pd.DataFrame(A.cpu().numpy())
#                 pos_arr_str = [str(s) for s in pos_arr]
#                 df_scores['pos'] = pos_arr_str
#                 df_scores.to_csv(os.path.join(args.score_path, slide_name+'.csv'), index=False)               
                
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    # select features
    parser.add_argument('--scale', type=int,help='select magnificant scale of the extracted patches,select form [10,20]')
    parser.add_argument('--extractor', type=str, default='none',help='select pretrained embedder')
    
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
#     parser.add_argument('--thres', nargs='+', type=float, default=[0.7371, 0.2752])
    parser.add_argument('--class_name', nargs='+', type=str, default=None)
    # parser.add_argument('--embedder_weights', type=str, default='test/weights/embedder.pth')
    # parser.add_argument('--aggregator_weights', type=str, default='test/weights/aggregator.pth')
    # parser.add_argument('--bag_path', type=str, default='test/patches')
    # parser.add_argument('--patch_ext', type=str, default='jpg')
    parser.add_argument('--map_path', type=str, default='attention_map')
    parser.add_argument('--export_scores', type=int, default=0)
    parser.add_argument('--score_path', type=str, default='score')

    # select the model
    parser.add_argument('--weight_path',help='path of best model')
    args = parser.parse_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    set_seed()    

    # load trained model
    milnet = load_model(args)
    if isinstance(milnet,torch.nn.DataParallel):
        milnet = milnet.module
    milnet.eval()

    # load data
    bags_path = f'data/TCGA_bladder_threshold_80/features/size512/{args.extractor}/{args.scale}X_features.pkl'
    with open(bags_path,'rb') as f:
        bags = pickle.load(f) # bags{dict}: 'slide_id':{'features','coords'}
        
    # create path for saving
    map_path = os.path.join('data/TCGA_bladder_threshold_80/visualization',args.map_path,Path(args.weight_path).stem)
    score_path = os.path.join('data/TCGA_bladder_threshold_80/visualization',args.score_path,Path(args.weight_path).stem)
    os.makedirs(map_path, exist_ok=True)
    if args.export_scores:
        os.makedirs(score_path, exist_ok=True)

    if args.class_name == None:
        args.class_name = ['TMB_L','TMB_H']
    test(args, bags, milnet)