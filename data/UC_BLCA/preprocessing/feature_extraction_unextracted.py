import os
from PIL import Image
from tqdm import tqdm
import argparse
import pickle

from timm.models.layers.helpers import to_2tuple
import timm
import torch
from torch import nn
from torchvision import models as torch_models
import torchvision.transforms as transforms
from skimage import io
import pandas as pd
import numpy as np
from pathlib import Path
import h5py

from model.resnet_simclr import ResNetSimCLR
from model.resnet_custom import resnet50_baseline
from retccl_extractor_dependencies import ResNet as ResNet


class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x




class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model
    
def model_loader(model,state_dict,args):
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=eval(args.gpu))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()} #model trained using nn.dataparallel, needs to rename the keys when apply to a new implemented model without using nn.DataParallel
    model.load_state_dict(state_dict)
    model.cuda()
    
def main():
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')

    '''general args to be assigned'''

    # presets, gpu index & batch_size
    parser.add_argument('--gpu', type=str, default='0', help='GPU device,eg. 0,1,2 [0]')
    parser.add_argument('--batch_size',type=int,default=512)

    # features to be extracted
    parser.add_argument('--scale',type=int,help='used when arg::extractor == res!  scale of slides to be extracted')

    # extractor type
    parser.add_argument('--extractor', type=str,help='choose the feature extractor')

    # default args
    parser.add_argument('--outdim',type=int,default=512,help='dim of extracted features, should be assigned in simclr extractor')
#     parser.add_argument('--tile_size',type=int,default=512,choices=[512,224])

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    #load the model
    if args.extractor == 'pretrained_resnet0':
        model = resnet50_baseline(pretrained=True)
        model.cuda()
    elif args.extractor == 'pretrained_resnet18':
        model = ResNet_extractor(layers=18) 
        model.cuda()
    elif args.extractor == 'pretrained_resnet50':
        model = ResNet_extractor(layers=50) 
        model.cuda()
    elif args.extractor == 'simclr':
#         state_dict = torch.load(f'simclr_feature_extractor/runs/{args.file_path}/checkpoints/model.pth')   #需要修改
        model = ResNetSimCLR(base_model=args.model,out_dim=args.outdim)
        model_loader(model,state_dict,args)
    elif args.extractor == 'retccl': # get features in shape N*2048
        model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        state_dict = torch.load(r'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/retccl_extractor_dependencies/best_ckpt.pth')
        model.fc = nn.Identity()
        model_loader(model,state_dict,args)
    elif args.extractor == 'cTransPath': # get features in shape N*768
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/retccl_extractor_dependencies/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
        model.cuda()
    else:
        model_path = {
            'TCGA_high':'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/simclr_feature_extractor/pretrained_embedder/TCGA/model-high-v1.pth',
            'TCGA_low':'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/simclr_feature_extractor/pretrained_embedder/TCGA/model-low-v1.pth',
            'c16_high':'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/simclr_feature_extractor/pretrained_embedder/c16/20X-model-v2.pth',
        }
        state_dict = torch.load(model_path[args.load])
        model = ResNetSimCLR(base_model=args.model,out_dim=args.outdim)
        model_loader(model,state_dict,args)
#     if len(args.gpu) > 1:
#         model = torch.nn.DataParallel(model, device_ids=eval(args.gpu))
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#         state_dict = {k.replace('module.',''):v for k,v in state_dict.items()} #model trained using nn.dataparallel, needs to rename the keys when apply to a new implemented model without using nn.DataParallel
    
#     model.load_state_dict(state_dict)
#     model = model.cuda()
    
    # load tile paths of slides
    data_transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Resize(224),          
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch_size = args.batch_size
    
    unextracted_slide_id = np.load(f'/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/features/size512/{args.extractor}/{args.scale}X_not_extracted.npy')
    patches = [Path(f'/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/CN_patches/size512/{args.scale}X/{i}.h5') for i in unextracted_slide_id]
    
    save_dir = '/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/features/size512/' + f'{args.extractor}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{args.scale}X_features.pkl'

    print(f'start to process,scale{args.scale}X')
    unextracted = []
    
    if args.extractor in ['pretrained_resnet0','pretrained_resnet18','pretrained_resnet50','retccl','cTransPath']:
        with torch.no_grad():
            model = model.eval()
            with open(save_path,'rb') as s:
                slide_features = pickle.load(s)
                for patch in tqdm(patches):
                    slide_id = patch.stem
                    seq_id = slide_id.split('_')[-1]
                    try:
                        with h5py.File(patch,'r') as f:
                            CN_patches = f['image_array'][:]
                            feature_list = []

                            N_tumor_patch = len(CN_patches)
                            
                            for batch_idx in range(0, N_tumor_patch, batch_size):
                                end = batch_idx + batch_size if batch_idx+batch_size < N_tumor_patch else N_tumor_patch
                                batch_patches = CN_patches[batch_idx: end]
                                images = []
                                for image in batch_patches:
                                    image_tensor = data_transform(image).unsqueeze(0)
                                    images.append(image_tensor)
                                images = torch.cat(images, dim=0)

                                features = model(images.cuda()) 
                                if len(features.shape) == 1:
                                    features = features.unsqueeze(0)
                                feature_list.append(features.detach().cpu())
                                del features

                            feature_list = torch.cat(feature_list, dim=0)

                            slide_features[slide_id] = {'seq_id':seq_id,'features': feature_list}

                    except Exception as e:
                        print(f'wrong in slide{slide_id}, error as {e}')
                        unextracted.append(slide_id)
                        np.save(os.path.join(save_dir,f'{args.scale}X_not_still_extracted.npy'),unextracted)
                pickle.dump(slide_features,s)
                
    else:
        with torch.no_grad():
            model = model.eval()
            with open(save_path,'rb') as s:
                slide_features = pickle.load(s)
                for patch in tqdm(patches):
                    slide_id = patch.stem
                    seq_id = slide_id.split('/')[-1]

                    try:
                        with h5py.File(patch,'r') as f:
                            CN_patches = f['image_array'][:]
                            feature_list = []

                            N_tumor_patch = len(CN_patches)
                            
                            for batch_idx in range(0, N_tumor_patch, batch_size):
                                end = batch_idx + batch_size if batch_idx+batch_size < N_tumor_patch else N_tumor_patch
                                batch_patches = CN_patches[batch_idx: end]
                                images = []
                                for patch in batch_patches:
                                    image_tensor = data_transform(patch).unsqueeze(0)
                                    images.append(image_tensor)
                                images = torch.cat(images, dim=0)

                                features = model(images.cuda())[1] # feature有问题！看一下res50/18 / retccl feature长度是否都很短
                                if len(features.shape) == 1:
                                    features = features.unsqueeze(0)
                                feature_list.append(features.detach().cpu())
                                del features

                            feature_list = torch.cat(feature_list, dim=0)

                            slide_features[slide_id] = {'seq_id':seq_id,'features': feature_list}

                    except Exception as e:
                        print(f'wrong in slide{slide_id}, error as {e}')
                        unextracted.append(slide_id)
                        np.save(os.path.join(save_dir,f'{args.scale}X_not_still_extracted.npy'),unextracted)
                pickle.dump(slide_features,s)
                
    print(f'{len(unextracted)} unextracted,saved in file',end = '\n')
    if len(unextracted) == 0:
        os.remove(f'/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/features/size512/{args.extractor}/{args.scale}X_not_extracted.npy')
    print(f'{args.extractor}/{args.scale}X_not_extracted.npy has been removed')
    print('file saved')

if __name__ == '__main__':
    main()