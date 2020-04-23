# coding:utf-8
import glob
import os.path
import numpy as np
import os
import re
import argparse
import cv2
'''
创建验证集bin的pairs.txt
'''
import random
# 图片数据文件夹

parser = argparse.ArgumentParser(description='generate image pairs')
parser.add_argument('--data-dir',type=str,default='F:/FaceData/lfw',help='')
parser.add_argument('--output-txt',type=str,default='F:/FaceData/lfw/pairs.txt',help='')
parser.add_argument('--num-pairs',type =int ,default=6000,help='')

args = parser.parse_args()

pairs_file_path = args.output_txt

rootdir_list = os.listdir(args.data_dir)
idsdir_list = [name for name in rootdir_list if os.path.isdir(os.path.join(args.data_dir, name))]
id_nums = len(idsdir_list)

def produce_same_pairs():
    matched_result = []  # 不同类的匹配对
    for j in range(args.num_pairs):
        id_int= random.randint(0,id_nums-1)

        the_img_name = idsdir_list[id_int]

        id_dir = os.path.join(args.data_dir,the_img_name)

        id_imgs_list = os.listdir(id_dir)

        id_list_len = len(id_imgs_list)

        id1_img_file = id_imgs_list[random.randint(0,id_list_len-1)]
        id2_img_file = id_imgs_list[random.randint(0,id_list_len-1)]

        id1_path = os.path.join(id_dir, id1_img_file)
        id2_path = os.path.join(id_dir, id2_img_file)

        same = 1
        #print([id1_path + '\t' + id2_path + '\t',same])
        matched_result.append((id1_path + '\t' + id2_path + '\t',same))
    return matched_result


def produce_unsame_pairs():
    unmatched_result = []  # 不同类的匹配对
    for j in range(args.num_pairs):
        id1_int = random.randint(0,id_nums-1)
        id2_int = random.randint(0,id_nums-1)
        while id1_int == id2_int:
            id1_int = random.randint(0,id_nums-1)
            id2_int = random.randint(0,id_nums-1)

        id1_img_name = idsdir_list[id1_int]
        id2_img_name = idsdir_list[id2_int]

        id1_dir = os.path.join(args.data_dir, id1_img_name)
        id2_dir = os.path.join(args.data_dir, id2_img_name)

        id1_imgs_list = os.listdir(id1_dir)
        id2_imgs_list = os.listdir(id2_dir)
        id1_list_len = len(id1_imgs_list)
        id2_list_len = len(id2_imgs_list)

        id1_img_file = id1_imgs_list[random.randint(0, id1_list_len-1)]
        id2_img_file = id2_imgs_list[random.randint(0, id2_list_len-1)]

        id1_path = os.path.join(id1_dir, id1_img_file)
        id2_path = os.path.join(id2_dir, id2_img_file)

        same = 0
        unmatched_result.append((id1_path + '\t' + id2_path + '\t',same))
    return unmatched_result


same_result = produce_same_pairs()
unsame_result = produce_unsame_pairs()

all_result = same_result + unsame_result

random.shuffle(all_result)
#print(all_result)

file = open(pairs_file_path, 'w')
for line in all_result:
    file.write(line[0] + str(line[1]) + '\n')

file.close()

import sys
import os

# path = 'F:/FaceData/lfw\Lee_Ann_Terlaji\Lee_Ann_Terlaji_0001.jpg'
# lfw_bins = []
# with open(path, 'rb') as fin:
#     _bin = fin.read()
#     lfw_bins.append(_bin)
# print(len(lfw_bins[0]))