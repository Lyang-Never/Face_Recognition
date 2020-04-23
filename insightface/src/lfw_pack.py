import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import numpy as np


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_paths(pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    flag = 1
    for pair in pairs:
        path0 = pair[0]
        path1 = pair[1]
        tmp = path0,path1,pair[2],type(pair[2])
        print(tmp)
        if flag == int(pair[2]):
            issame = True
        else:
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list


parser = argparse.ArgumentParser(description='Package LFW images')

parser.add_argument('--data-dir', default='F:/FaceData/lfw1/pairs.txt', help='')
parser.add_argument('--image-size', type=str, default='112,112', help='')
parser.add_argument('--output', default='F:/FaceData/lfw1/lfw.bin', help='path to save.')
args = parser.parse_args()

lfw_dir = args.data_dir
image_size = [int(x) for x in args.image_size.split(',')]
lfw_pairs = read_pairs(lfw_dir)
lfw_paths, issame_list = get_paths(lfw_pairs)

lfw_bins = []
#lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
i = 0
for path in lfw_paths:
  with open(path, 'rb') as fin:
    _bin = fin.read()
    lfw_bins.append(_bin)
    #img = mx.image.imdecode(_bin)
    #img = nd.transpose(img, axes=(2, 0, 1))
    #lfw_data[i][:] = img
    i+=1
    if i%1000==0:
      print('loading lfw', i)

with open(args.output, 'wb') as f:
  pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
