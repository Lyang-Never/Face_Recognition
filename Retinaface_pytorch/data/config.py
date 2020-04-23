# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'min_sizes1': [[32, 64, 128], [256], [512]],
    'steps1': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

from easydict import EasyDict as edict

test_fddb_args = edict()

test_fddb_args.trained_model = './weights/mobilenet0.25_Final.pth'
test_fddb_args.network = 'mobile0.25'
test_fddb_args.save_folder = 'eval/'
test_fddb_args.cpu = False
test_fddb_args.dataset = 'FDDB'
test_fddb_args.confidence_threshold = 0.02
test_fddb_args.top_k = 5000
test_fddb_args.nms_threshold = 0.4
test_fddb_args.keep_top_k = 750
test_fddb_args.save_image = True
test_fddb_args.vis_thres = 0.5
