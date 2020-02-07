import numpy as np
import mxnet as mx
import argparse
import os
import pickle
import cv2
from core.symbol import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector
from prepare_data.utils import IoU  # 直接import 类
from prepare_data.utils import convert_to_square


def test_net(root_path, dataset_path, image_set, prefix, epoch,
             batch_size, ctx, test_mode="onet",
             thresh=[0.6, 0.6, 0.7], min_face_size=24,
             stride=2, slide_window=False, shuffle=False, vis=False):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        args, auxs = load_param(prefix[1], epoch[1], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    imdb = IMDB("fddb", image_set, root_path, dataset_path, 'test')
    gt_imdb = imdb.gt_imdb()

    test_data = TestLoader(gt_imdb)
    print('1')
    # 得到的detections格式为:[[第一张图的所有预测框],[第二张图片的所有预测框],[第三张图片的所有预测框],...,[最后一张图片的所有预测框]],每个预测框为[x1,y1,x2,y2,score],score为该预测框为人脸的分数
    detections = mtcnn_detector.detect_face(imdb, test_data, vis=vis)
    save_path = "/Users/qiuxiaocong/Downloads/mtcnn1"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections_onet_0009_givenPRnet.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, pickle.HIGHEST_PROTOCOL)
    print('detections saved done!')

    # if test_mode == "pnet":
    #     net = "rnet"
    # elif test_mode == "rnet":
    #     net = "onet"
    #
    # save_path = "/Users/qiuxiaocong/Downloads/mtcnn1/prepare_data/%s"%net
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # save_file = os.path.join(save_path, "detections.pkl")
    # with open(save_file, 'wb') as f:
    #     pickle.dump(detections, f, pickle.HIGHEST_PROTOCOL)
    # save_hard_example(net)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='/Users/qiuxiaocong/Downloads', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='/Users/qiuxiaocong/Downloads/mtcnn1', type=str)
    parser.add_argument('--image_set', dest='image_set', help='image set',
                        default='FDDB-fold-all', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['model/pnet', 'model/rnet', 'model/onet'], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 9], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.7, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=-1, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        ctx = mx.cpu(0)
    test_net(args.root_path, args.dataset_path, args.image_set, args.prefix,
             args.epoch, args.batch_size, ctx, args.test_mode,
             args.thresh, args.min_face, args.stride,
             args.slide_window, args.shuffle, args.vis)

