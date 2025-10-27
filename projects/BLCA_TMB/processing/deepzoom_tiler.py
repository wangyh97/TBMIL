import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize

import pandas as pd
import cv2
from lxml import etree
from pathlib import Path
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'

def AnnotationParser(path):
    assert Path(path).exists(), "This annotation file does not exist."
    tree = etree.parse(path)
    annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
    annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
    classes = [group.attrib["Name"] for group in annotation_groups]

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
    if "stroma" not in classes:
        masks["stroma"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)

    return masks 

def get_mask_slide(masks):
    tumor_slide = openslide.ImageSlide(Image.fromarray(masks['tumor']))
    return tumor_slide

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        contour_path = Path(self._slidepath).with_suffix('.xml') 
        rule = {"tumor":{"excludes":["artificial","stroma","necrosis"]},
            'stroma':{"excludes":['artificial','necrosis']}}
        contour = Annotation(self._slide,path=str(contour_path),rule= rule)

        self._mask = get_mask_slide(contour) # get mask of the slide


        last_associated = None
        dz,mask_dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                tumor_tile = mask_dz.get_tile(level, address)
                percent_unmasked = np.mean(tumor_tile)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size
                if all(edge > self._threshold, percent_unmasked>0.75):
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,limit_bounds=self._limit_bounds),\
                DeepZoomGenerator(self._mask,self._tile_size,self._overlap,limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        print('run tiler')
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)  # all 40 here
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0,if 0,1  & set base_mag = 20, target_levels = [1,2]
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, output_patches, level=(0,), ext='tiff'):
    print('\n Organizing patches')
    img_name = img_slide.split(os.sep)[-1].split('.')[0]
    img_class = img_slide.split(os.sep)[2]
    # n_levels = len(glob.glob('WSI_temp_files/*'))
    bag_path = os.path.join(output_patches, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    if len(level)==1:
        patches = glob.glob(os.path.join('WSI_temp_files', '*', '*.'+ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join('WSI_temp_files', '*'))]
        levels.sort()
        low_patches = glob.glob(os.path.join('WSI_temp_files', str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join('WSI_temp_files', str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')

if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    # parser.add_argument('-d', '--dataset', type=str, default='TCGA-lung', help='D ataset name')
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='tiff', help='Image format for tiles [tiff]')
    parser.add_argument('-v', '--slide_format', type=str, default='svs', help='Image format for tiles [svs]')
    parser.add_argument('-j', '--workers', type=int, default=12, help='Number of worker processes to start [4]')
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
    parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]')
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0,), help='Levels for patch extraction [0]')
    parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
    parser.add_argument('-t', '--background_t', type=int, default=25, help='Threshold for filtering background [25]')  
    args = parser.parse_args()
    levels = tuple(args.magnifications)

    print(len(levels))

    assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'

    # set input & output paths
    ## input paths
    all_slides = pd.read_csv('project/BLCA_TMB/config/full.csv')['svs_paths'][3:4]
    ## output paths
    if len(levels) == 2:
        output_patches = os.path.join('data/raw_patches/size_224', 'pyramid')
    else:
        output_patches = os.path.join('data/raw_patches/size_224', 'single')
    if not os.path.exists(output_patches):
        os.makedirs(output_patches)

    print(f'read input from:{all_slides}, output to:{output_patches}')
    
    # pos-i_pos-j -> x, y
    for idx, c_slide in enumerate(all_slides):
        print('Process slide {}/{}'.format(idx+1, len(all_slides)))
        DeepZoomStaticTiler(c_slide, 'WSI_temp', levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
        nested_patches(c_slide, output_patches, levels, ext=args.format)
        shutil.rmtree('WSI_temp_files') 
    print('Patch extraction done for {} slides.'.format(len(all_slides)))