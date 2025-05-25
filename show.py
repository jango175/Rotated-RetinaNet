from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import numpy as np
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results
from eval import evaluate
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge


def main(args):
    if os.path.exists('outputs/dota_out'):
        shutil.rmtree('outputs/dota_out')
    os.mkdir('outputs/dota_out')

    if os.path.exists('dota_res'):
        shutil.rmtree('dota_res')
    os.mkdir('dota_res')

    # exec('cd outputs &&  rm -rf detections && rm -rf integrated  && rm -rf merged')    
    ResultMerge('outputs/detections', 
                'outputs/integrated',
                'outputs/merged',
                'outputs/dota_out')
    img_path = os.path.join(args.ims_dir,'images')
    label_path = 'outputs/dota_out'
    save_imgs = True
    if save_imgs:
        show_dota_results(img_path,label_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--ims_dir', type=str, default='DOTA/test')
    main(parser.parse_args())