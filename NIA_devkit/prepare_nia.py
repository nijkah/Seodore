import utils as util
import os
import imgsplit_multiprocess
import split_onlyimage_multiprocess
import shutil
from multiprocessing import Pool
from geojson2coco import geojson2coco
import argparse

CLASS_NAMES_EN = ('background', 'small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car', 'bus', 'truck', 'train',
        'crane', 'bridge', 'oil tank', 'dam', 'athletic field', 'helipad', 'roundabout')

def parse_args():
    parser = argparse.ArgumentParser(description='prepare nia_for_split')
    parser.add_argument('--srcpath', default='../data/nia_1200')
    parser.add_argument('--dstpath', default=r'../data/nia-split-512',
                        help='prepare data')
    args = parser.parse_args()

    return args



def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> train512, test --> test512
    :return:
    """
    rate = 2
    train_dst_name = 'train768_2x_1200'
    test_dst_name = 'test1024_2x'

    if not os.path.exists(os.path.join(dstpath, test_dst_name)):
        os.makedirs(os.path.join(dstpath, test_dst_name))
    if not os.path.exists(os.path.join(dstpath, train_dst_name)):
        os.makedirs(os.path.join(dstpath, train_dst_name))

    split_train = imgsplit_multiprocess.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, train_dst_name),
                      gap=256,
                      subsize=768,
                      num_process=32
                      )
    split_train.splitdata(rate)

    # split_val = imgsplit_multiprocess.splitbase(os.path.join(srcpath, 'val'),
    #                    os.path.join(dstpath, 'trainval1024'),
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_val.splitdata(1)

    #split_test = split_onlyimage_multiprocess.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                   os.path.join(dstpath, test_dst_name, 'images'),
    #                  gap=256,
    #                  subsize=1024,
    #                  num_process=32
    #                  )
    #split_test.splitdata(rate)

    geojson2coco(os.path.join(dstpath, train_dst_name, 'images'), os.path.join(dstpath, train_dst_name, 'json'), os.path.join(dstpath, train_dst_name, 'coco.json'))
    #geojson2coco(os.path.join(dstpath, test_dst_name, 'images'), os.path.join(dstpath, test_dst_name, 'json'), os.path.join(dstpath, test_dst_name, 'coco.json'))

if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
