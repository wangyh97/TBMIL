#!/usr/bin/env python
# coding: utf-8

# In[4]:

#!/usr/bin/env python
# coding: utf-8

# In[4]:
# from numba import jit,njit

import os
from pathlib import Path
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import skimage
from lxml import etree
from matplotlib import pyplot as plt

from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gc
import time
import sys
import getopt
import pandas as pd

def get_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide
# 提高亮度，处理异常像素点的函数
def normalize_dynamic_range(image, percentile = 95):
    """
    Normalize the dynamic range of an RGB image to 0~255. If the dynamic ranges of patches 
    from a dataset differ, apply this function before feeding images to VahadaneNormalizer,
    e.g. hema slides.
    :param image: A RGB image in np.ndarray with the shape [..., 3].
    :param percentile: Percentile to get the max value.
    """
    max_rgb = []
    for i in range(3):
        value_max = np.percentile(image[..., i], percentile)
        max_rgb.append(value_max)
    max_rgb = np.array(max_rgb)

    new_image = (np.minimum(image.astype(np.float32) * (255.0 / max_rgb), 255.0)).astype(np.uint8)
    
    return new_image

# 定义过滤白色的函数
def filter_blank(image, threshold = 100): # threshold = 80有点小了，调到100比较合适
    image_lab = skimage.color.rgb2lab(np.array(image)) #将图像由RGB转为LAB
    image_mask = np.zeros(image.shape).astype(np.uint8) #制造mask
    image_mask[np.where(image_lab[:, :, 0] < threshold)] = 1 
    image_filter = np.multiply(image, image_mask)
    percent = ((image_filter != np.array([0,0,0])).astype(float).sum(axis=2) != 0).sum() / (image_filter.shape[0]**2)

    return percent

#@jit(nopython=True)
def AnnotationParser(path):
    assert Path(path).exists(), "This annotation file does not exist."
    tree = etree.parse(path)
    annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
    annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
    classes = [group.attrib["Name"] for group in annotation_groups]
   # @jit(nopython=True)
    def read_mask_coord(cls):
        for annotation in annotations:
            if annotation.attrib["PartOfGroup"] == cls:
                contour = []
                for coord in annotation.xpath("Coordinates/Coordinate"):
                    x = np.float(coord.attrib["X"])
                    y = np.float(coord.attrib["Y"])
                    contour.append([round(float(x)),round(float(y))])
                #mask_coords[cls].extend(contour)
                mask_coords[cls].append(contour)
    #@jit(nopython=True)
    def read_mask_coords(classes):
        for cls in classes:
            read_mask_coord(cls)
        return mask_coords            
    mask_coords = {}
    for cls in classes:
        mask_coords[cls] = []
    mask_coords = read_mask_coords(classes)
    return mask_coords,classes


def Annotation(slide,path,save_path=None,rule=False,save=False):
    #wsi_height = slide.wsi_height
    #wsi_width = slide.wsi_width
    wsi_width,wsi_height = slide.level_dimensions[0]
    masks = {}
    contours = {}
    mask_coords, classes = AnnotationParser(path)
    
    def base_mask(cls,wsi_height,wsi_width):
        masks[cls] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    def base_masks(wsi_height,wsi_width):
        for cls in classes:
            base_mask(cls,wsi_height,wsi_width)
        return masks
    
    def main_masks(classes,mask_coords,masks):
        for cls in classes:
            contours = np.array(mask_coords[cls])
            #contours = mask_coords[cls]
            for contour in contours:
                #print(f"cls:{cls},\ncontour:{contour},\ntype:{type(contour)}")
                masks[cls] = cv2.drawContours(masks[cls],[np.int32(contour)],0,True,thickness=cv2.FILLED)
        return masks
   # def make_masks()
    def export_mask(save_path,cls):
        assert Path(save_path).is_dir()
        cv2.imwrite(str(Path(save_path)/"{}.tiff".format(cls)),masks[cls],(cv2.IMWRITE_PXM_BINARY,1))
    def export_masks(save_path):
        for cls in masks.keys():
            export_mask(save_path,cls)
            
    def exclude_masks(masks,rule,classes):
        #masks_exclude = masks.copy()
        masks_exclude = masks
        for cls in classes:
            for exclude in rule[cls]["excludes"]:
                if exclude in masks:
                    overlap_area = cv2.bitwise_and(masks[cls],masks[exclude])
                    masks_exclude[cls] = cv2.bitwise_xor(masks[cls],overlap_area)
        #masks = masks_exclude
        return masks_exclude
                    
    masks = base_masks(wsi_height,wsi_width)
    masks = main_masks(classes,mask_coords,masks)
    if rule:
        classes = list(set(classes) & set(rule.keys()))
        masks = exclude_masks(masks,rule,classes)
        #include_masks(rule)
        #exclude_masks(rule)
    if save:
        export_masks(save_path)

    
    if "artificial" not in classes:
        masks["artificial"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if "necrosis" not in classes:
        masks["necrosis"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8) 
        #TODO:这里要不要stroma？如何识别全片区域  
    if "stroma" not in classes:
        masks["stroma"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)

    return masks 

    if "artificial" not in classes:
        masks["artificial"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if "necrosis" not in classes:
        masks["necrosis"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if 'stroma' not in classes:
        masks['stroma'] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    return masks


def show_thumb_mask(mask,size=512):
    #mask = masks[cls]
    height, width = mask.shape
    scale = max(size / height, size / width)
    mask_resized = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    mask_scaled = mask_resized * 255
    plt.imshow(mask_scaled)
    return mask_scaled

def get_mask_slide(masks):
    tumor_slide = openslide.ImageSlide(Image.fromarray(masks['tumor']))
    non_tumor_slide = openslide.ImageSlide(Image.fromarray(cv2.bitwise_not(masks['tumor'])-254))
    #mark_slide = openslide.ImageSlide(Image.fromarray(masks["mark"])) ## get tile_masked dont need mark and arti mask
    #arti_slide = openslide.ImageSlide(Image.fromarray(masks["artifact"]))
    return (tumor_slide,non_tumor_slide)

def get_tiles(slide,tumor_slide,tile_size=512,overlap=False,limit_bounds=False,slide_tile = False):
    slide_tiles = DeepZoomGenerator(slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    tumor_tiles = DeepZoomGenerator(tumor_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    #mark_tiles = DeepZoomGenerator(mark_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    #arti_tiles = DeepZoomGenerator(arti_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    if slide_tile:
        return slide_tiles,tumor_tiles
    else:
        return tumor_tiles
#@njit
def remove_arti_and_mask(slide_tile,tumor_tile):
    #mark_tile = np.where(mark_tile==0,1,0)
    #arti_tile = np.where(arti_tile==0,1,0)
    #assert slide_tile.shape
    #slide_tile = np.array(slide_tile)
    #tumor_tile = np.array(tumor_tile)
    x = slide_tile.shape
    if not x == tumor_tile.shape:
        tumor_tile = tumor_tile[:x[0],:x[1],:]
    #if not mark_tile.shape == x:
       # mark_tile = mark_tile[:x[0],:x[1],:]
    #if not arti_tile.shape == x:
       # arti_tile = arti_tile[:x[0],:x[1],:]
    #tile = np.multiply(np.multiply(slide_tile,mark_tile),arti_tile)
    #if tile[np.where(tile==np.array([0,0,0]))].shape!=(0,):
        #tile[np.where(tile==np.array([0,0,0]))]= fill
    #tile[np.where(tile==np.array([0,0,0]))] = fill # fill blank may cause color torsion
    tile_masked= np.multiply(slide_tile,tumor_tile)
    #tile = Image.fromarray(np.uint8(tile))
    #assert tile.size==(512,512),f"wrong shape:{tile.size}"
    return slide_tile,tile_masked
def get_tile_masked(slide_tile,tumor_tile): ####version_update: To save tile_masked, use this function
    x = slide_tile.shape
    y = tumor_tile.shape
    if not x == y:
        h = np.min([x[0],y[0]])
        w = np.min([x[1],y[1]])
        tumor_tile = tumor_tile[:h,:w,:]
        slide_tile = slide_tile[:h,:w,:]
    tile_masked = np.multiply(slide_tile,tumor_tile)
    percent = np.mean(tumor_tile)
    tile_masked[np.all(tile_masked==0)]=255
    return tile_masked,percent
def filtered_same(img):### modify to purely count tumor tile
    percent = ((img[:,:,0]==img[:,:,1]).astype(float) *(img[:,:,0]==img[:,:,2]).astype(float)).sum()/(img.shape[0]**2)
    return percent
def filtered(tile):
    tolerance = np.array([230,230,230])
    #tile_1 = tile.copy()
    tile[np.all(tile>tolerance,axis=2)]=0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent
def filtered_cv(img):
    #tolerance = np.array([230,230,230])
    #tile_1 = tile.copy()
    tile = np.copy(img).astype(np.uint8)
    gray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,_ = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tile[np.all(tile>ret,axis=2)] = 0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent

def filter_blood(img):
    ## lower mask(0-10)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1
    percent = ((mask != 0)).sum()/mask.shape[0]**2
    return percent
#@jit(nopython=True)
def extract_patches(levels,scales,tile_path,slide_tiles,tumor_tiles,tumor=True):
    
    for i,level in enumerate(levels):
        if tumor:
            print(f'processing ---level {scales[i]},tumor tiles')
        else:
            print(f'processing ---level {scales[i]},non-tumor tiles')
        print(tile_path)
        tiledir = Path(tile_path)/str(scales[i])
        #print(f"tile_dir creating--{tiledir}")
        
        if not Path(tiledir).exists():
            os.makedirs(tiledir)
           # print("tile_dir created")
        assert slide_tiles.level_tiles[level] == tumor_tiles.level_tiles[level]
        cols,rows = slide_tiles.level_tiles[level]
        for row in range(rows):
            for col in range(cols):
                if tumor:
                    tilename = os.path.join(tiledir,'%s_%d_%d.%s'%('T',col,row,"tiff"))
                else:
                    tilename = os.path.join(tiledir,'%s_%d_%d.%s'%('nonT',col,row,"tiff"))
               # print("tile_name creating")
                if not Path(tilename).exists():
                    slide_tile = np.array(slide_tiles.get_tile(level,(col,row)))
                    tumor_tile = np.array(tumor_tiles.get_tile(level,(col,row)))
                    #mark_tile = np.array(mark_tiles.get_tile(level,(col,row)))
                    #arti_tile = np.array(arti_tiles.get_tile(level,(col,row)))
                    #print("tiles are processing")
                    #tile,tile_masked = remove_arti_and_mask(slide_tile,tumor_tile,mark_tile,arti_tile)
                    tile_masked,percent_2 = get_tile_masked(slide_tile,tumor_tile) # percent of annotated area       
                   # tile_masked = np.multiply(slide_tile,mark_tile)
                    percent_1 = filter_blank(tile_masked) # percent of tissue area, 当标记良好，没有留下大片空白区域时，可以不用这个,如果用，则用100作为threshold
                    #percent_2 = filtered_same(tile_masked)
                  #  percent_3 = filter_blood(tile_masked)

                    if all((percent_1 >= 0.75,percent_2 >= 0.75)) and tile_masked.shape==(512,512,3): # 如果tile_masked形状不一致np.array转化后的dtype为object，无法使用h5保存
#                     if percent_2 >= 0.75:
                       # Image.fromarray(np.uint8(tile)).save(tilename)
                        Image.fromarray(np.uint8(tile_masked)).save(tilename)
                        #print("saving tile")
                    else:
                        pass
        print("Done!")
    print("All levels processed!!")


if __name__ == '__main__':
    working_dir = 'data/'
    os.chdir(working_dir)

    # TCGA 388

    argv = sys.argv[1:]

    INDEX= 0
    n = 5

    try:
        opts,args = getopt.getopt(argv,"s:n:i:x:")
    except:
        print("Error")
    for opt,arg in opts:
        if opt in ['-n']: #代表chunksize
            n = int(arg)  #n后面承接一个数值
        elif opt in ['-s']:
            subset = arg  #s后面承接的是subsets，这里的是TCGA_BLCA及TCGA_BLCA_S两个分开的数据集
      #  elif opt in ['-i']:
           # i = int(arg)  #i后面承接一个数值
        elif opt in ['-i']: ## 用INDEX 表示每个chunk的序数
            INDEX = int(arg)  #x后面承接一个数值,表示index


    OVERLAP =0
    LIMIT = False
    rule =  {"tumor":{"excludes":["artificial","stroma","necrosis"]},
            'stroma':{"excludes":['artificial','necrosis']}}
    scales = ['10X','20X']
    TILE_SIZE = 512

    #TODO:将svspath及svslabels存放在一个csv中
    df = pd.read_csv('project/BLCA_TMB/config/full.csv')   # well marked slides & slides with label的交集
    svs_paths = df['svs_paths']
    labels = df['TMB_H/L']
    uuid = df['dir_uuid']

    svs_paths = svs_paths[n*(INDEX-1):n*INDEX]
    patch_path = "data/raw_patches"



    extracted_case = []
    un_extracted_case = []
    for i,svs in enumerate(svs_paths):  #svs是一个svs图像路径的str
        start = time.time()
        total_num = len(svs_paths)
        print(f"processing  {i+1}/{total_num}:------{svs}")

        #路径操作
       # label = labels[i]
        label = df.loc[df['svs_paths']==svs]['TMB_H/L'].to_list()[0] ## 在这里用svs_path来取label的值
        xml_path = Path(svs).with_suffix('.xml')   #返回一个path
        #构造存放patch的目录，目录的结构为
        case_name = df.loc[df['svs_paths']==svs]['dir_uuid'].to_list()[0]
    #     case_name = Path(svs).parent.name
        tile_path = Path(patch_path)/label/case_name

        #提取操作
        slide = get_slide(str(svs))
        try:
            masks = Annotation(slide,path=str(xml_path))
            print(f"masks groups includes :{list(masks.keys())}")
            tumor_XOR = get_mask_slide(masks)    #返回一个tuple，第一个是tumor_slide，第二个是non_tumor_slide，两个都是Imageslide

            #获得dzg对象                                      
            slide_tiles,tumor_tiles = get_tiles(slide,tumor_XOR[0],tile_size=TILE_SIZE,overlap=OVERLAP,limit_bounds=LIMIT,slide_tile=True)
#             non_tumor_tiles = get_tiles(slide,tumor_XOR[1],tile_size=TILE_SIZE,overlap=OVERLAP,limit_bounds=LIMIT)

            del slide
            del masks
            del tumor_XOR
            gc.collect()
            level_count = slide_tiles.level_count
            #fill = int(np.array(slide_tiles.get_tile(level_count-1,(0,0))).mean())
            levels=[level_count-3,level_count-2]   # 10,20
            #print(f"fill_blank_value:{fill}")

            try:
                extract_patches(levels,scales,tile_path,slide_tiles,tumor_tiles)
    #             extract_patches(levels,scales,tile_path,slide_tiles,non_tumor_tiles,tumor=False)
                extracted_case.append(svs)
            except Exception as e:
                un_extracted_case.append(svs)
                print("something is wrong when extracting")
                print("ERROR!",e)
                continue
        except Exception as e:
            print("something is wrong when parsing")
            print("ERROR!",e)
            continue
        end = time.time()
        print(f"Time consumed : {(end-start)/60} min")
        print(f"******{len(un_extracted_case)}/{len(svs_paths)} remain unextract******")


#unextracted cases:
#1:f4ca3ddd-dc53-4ab0-b55b-942603b64e57