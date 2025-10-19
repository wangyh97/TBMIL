import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import time
from PIL import Image
from torchvision import transforms, datasets
from imblearn.combine import SMOTETomek
from multiprocess import Pool,set_start_method
from tqdm import tqdm
from random import shuffle
import gc
import os

class dataset_reform():
    def __init__(self,dataset):
        ''' 
        args:
            dataset:path of a npy file,containing a dict, arranged in form:
                {'train_list':<a dataframe>,
                'val_list':<a dataframe>,
                'test_list':<a dataframe>
                },
            formed in dataset segmentation.ipynb
        '''
        data = np.load(dataset,allow_pickle=True).item()

        self.training = data['train_list']
        self.val = data['val_list']
        self.testing = data['test_list']

    def __getattribute__(self, name: str):
        return object.__getattribute__(self,name)
    
print('start!')
dataset_5x = dataset_reform('../config/data_segmentation_csv/5X_grouping.npy')
train_df = dataset_5x.training

img_path_list = sum(train_df['img_list'],[])
label_list = sum([list(train_df['TMB_H/L'].iloc[i])*train_df['5x'].iloc[i] for i in range(train_df.shape[0])],[])
print('train_df loading complete')


data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_and_transform_image(img_path):
    with Image.open(img_path) as img:
        try:
            if data_transforms is not None:
                img = data_transforms(img)
        except Exception as e:
            print(f'Cannot transform image {img_path}: {e}')
            return None
        return img
    
# def resample(tup):
#     img,lab = tup
#     images_stacked = list(img)
#     label_list_seg = list(lab)
    
#     smoteenn = SMOTETomek(sampling_strategy='not majority',random_state = 10)
#     resampled_images, resampled_labels = smoteenn.fit_resample(images_stacked, label_list_seg)
#     return resampled_images, resampled_labels

# def batch_resample(images_stacked,label_list):
#     for i in range(images_stacked.shape[0]//1000+1):
#         if i != range(images_stacked.shape[0]//1000+1)[-1]:
#             yield images_stacked[i*1000:(i+1)*1000],label_list[i*1000:(i+1)*1000]
#         elif i == images_stacked.shape[0]//1000:
#             yield images_stacked[i*1000:],label_list[i*1000:]
            


if __name__ == '__main__':
    img_list = []
    with Pool(24) as p:
        for img in p.imap(load_and_transform_image,tqdm(img_path_list)):
            img_list.append(img)
    p.close()
    p.join()
    print('imgs loaded & transformed',end='\n')
    
    #convert img_list to np.array
    images_np = list(map(np.array, tqdm(img_list, desc='images_np')))
    
    # Reshape the images to 2D arrays
    images_2d = [img.reshape(-1, img.shape[0] * img.shape[1] * img.shape[2]) for img in tqdm(images_np,desc='images_2d')]

    #shuffle them all 
    combined = list(zip(images_2d,label_list))
    shuffle(combined)
    images_2d[:],label_list[:] = zip(*combined)

    # Stack the 2D arrays into a single 2D array
    
    images_stacked = np.vstack(images_2d)
    
    del img_path_list
    del images_np
    del images_2d
    gc.collect()
    
    
    #resampling the images
#     resampled_images = []
#     resampled_labels = []
#     batch_iter = batch_resample(images_stacked,label_list)
#     batch_list = [i for i in batch_iter]
    
    
    tock = time.time()
#     with Pool() as p:
#         for batch_resampled_images,batch_resampled_labels in p.imap(resample,tqdm(batch_list)):
#             resampled_images.append(batch_resampled_images)
#             resampled_labels.extend(batch_resampled_labels)
#     p.close()
#     p.join()

#     for batch_resampled_images,batch_resampled_labels in list(map(resample,tqdm(batch_list))):
#         resampled_images.append(batch_resampled_images)
#         resampled_labels.extend(batch_resampled_labels)
    smoteenn = SMOTETomek(sampling_strategy='not majority',random_state = 10)
    resampled_images, resampled_labels = smoteenn.fit_resample(images_stacked, label_list)
    print(f'resampling consumes{time.time()-tock}s',end='\n')
    print('resample finished')
    
    save_dir = '../config/resample/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    np.save(f'{save_dir}SMOTETomek_imgs_2d_ls.npy',resampled_images)
    np.save(f'{save_dir}SMOTETomek_labels_ls.npy',resampled_labels)
    
    print('resampled imgs & labels saved')
    
    