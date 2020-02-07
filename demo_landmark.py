# -*- coding: UTF-8 -*-
import numpy as np
import mxnet as mx
import argparse
import cv2
import time
from core import *
from core.symbol import P_Net, R_Net, O_Net, L_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector1
from tools.image_processing import transform
import numpy as np
import mxnet as mx
import argparse
import cv2
import time
from core import *


def test_net(prefix, epoch, batch_size, ctx, thresh=[0.6, 0.6, 0.7], min_face_size=24,
                    stride=2, slide_window=False):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:   # 使用滑动窗口(MTCNN的P_NET不使用了滑动窗口,而是全卷积网络)
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[1], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector1(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                    stride=stride, threshold=thresh, slide_window=slide_window)

    # img = cv2.imread('test01.jpg')  # 读取图片
    # img = cv2.imread('zhang.jpeg')  # 读取图片
    # img = cv2.imread('curry.jpg')  # 读取图片
    # img = cv2.imread('physics.jpg')  # 读取图片
    # img = cv2.imread('000007.jpg')  # 读取图片
    # img = cv2.imread('test01.jpg')  # 读取图片
    # img = cv2.imread('NBA98.jpg')
    # img = cv2.imread('download.jpg')
    # img = cv2.imread('/Users/qiuxiaocong/Downloads/WIDER_train/images/7--Cheering/7_Cheering_Cheering_7_16.jpg')
    # img = cv2.imread('/Users/qiuxiaocong/Downloads/WIDER_train/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_77.jpg')
    # img = cv2.imread('/Users/qiuxiaocong/Downloads/3Dfacedeblurring/dataset_test/falling1/input/00136.png')
    img = cv2.imread('/Users/qiuxiaocong/Downloads/facetrack_python/error.jpg')

    boxes, boxes_c = mtcnn_detector.detect_pnet(img)
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)
    # print(boxes_c)  # x1 y1 x2 y2

    original_detect = []
    crop_list = []
    detect_len_list = []
    nd_array = []
    score_list = []

    if boxes_c is not None:
        draw = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX   # Python 一种字体
        idx = 0

        for b in boxes_c:     # nms和bbr之后的结果
            # 在draw上绘制矩形框(左上角坐标+右下角坐标)
            b_new0 = np.array(b[0:4])               # 添加检测框
            original_detect.append(b_new0)
            b_new = convert_to_square(b_new0)       # 添加送入到landmark net的48*48大小的框
            crop_list.append(b_new)
            score_list.append(b[4])

            # cv2.rectangle(draw, (int(b_new[0]), int(b_new[1])), (int(b_new[2]), int(b_new[3])),
            #               (0, 255, 255), 1)
            # # 在draw上添加文字
            # cv2.putText(draw, '%.3f'%b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)
            # cv2.imshow("detection result", draw)

            img_draw = img[int(b_new[1]):int(b_new[3]), int(b_new[0]):int(b_new[2])]
            detect_len = min(img_draw.shape[0],img_draw.shape[1])
            # print(img_draw.shape[0], img_draw.shape[1])
            if detect_len != 0:
                detect_len_list.append(detect_len)

                img_resized = cv2.resize(img_draw, (48, 48))
            # cv2.imshow("detection result", draw)
            # print('img_resized type is :{}'.format(type(img_resized)))
                nd_array.append(img_resized)

            # cv2.imwrite("detection_result{}.jpg".format(idx), img_resized)
            # cv2.waitKey(0)
                idx = idx + 1

    return crop_list, detect_len_list, original_detect, idx, img, nd_array


def test_landmark_net(crop_list, detect_len_list, original_detect, idx, img0, img_array):
    sym = L_Net('test')
    ctx = mx.cpu()

    # cv2.imshow("progin", img0)
    # load lnet model
    args, auxs = load_param('model/lnet', 4390, convert=False, ctx=ctx)  # 1990 3330 4390

    data_size = 48      # landmark net 输入的图像尺寸为48*48
    imshow_size = 48    # imshow_size为landmark结果展示的图片尺寸

    data_shapes = {'data': (1, 3, data_size, data_size)}
    disp_landmarks = []

    for idx_ in range(idx):
        # img = cv2.imread('./detection_result{}.jpg'.format(idx_))
        img = img_array[idx_]
        # img = cv2.resize(img, (data_size, data_size)) # 输入lnet的图片已经是48*48 无需resize
        # cv2.imshow("landmarks_10", img)
        # cv2.waitKey(0)
        newimg = transform(img)
        args['data'] = mx.nd.array(newimg, ctx)
        executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))
        # mx.cpu(), x=(5,4), grad_req='null'
        executor.copy_params_from(args, auxs)
        # print(executor.outputs)

        out_list = [[] for _ in range(len(executor.outputs))]
        executor.forward(is_train=False)
        for o_list, o_nd in zip(out_list, executor.outputs):
            o_list.append(o_nd.asnumpy())
        out = list()
        for o in out_list:
            out.append(np.vstack(o))
        landmarks = out[0]

        for j in range(int(len(landmarks)/2)):
            if landmarks[2 * j] > 1:
                landmarks[2 * j] = 1
            if landmarks[2 * j] < 0:
                landmarks[2 * j] = 0
            if landmarks[2 * j + 1] > 1:
                landmarks[2 * j + 1] = 1
            if landmarks[2 * j + 1] < 0:
                landmarks[2 * j + 1] = 0

        # print(len(landmarks))
        # print(landmarks)

        imshow_img = cv2.resize(img, (imshow_size, imshow_size))
        landmarks = landmarks * imshow_size
        # print('------------')
        # print(landmarks)
        # print('------------')
        landmarks = np.reshape(landmarks, -1)

        # for j in range(int(len(landmarks)/2)):
        #     cv2.circle(imshow_img, (int(landmarks[j]), (int(landmarks[j + 5]))), 2, (0, 0, 255),-1)
        # cv2.imshow("landmarks_10", imshow_img)
        # cv2.waitKey(0)

        fator = detect_len_list[idx_]/48.0
        disp_landmark = []

        for j in range(int(len(landmarks) / 2)):
            display_landmark_x = int(landmarks[j] * fator + crop_list[idx_][0])
            display_landmark_y = int(landmarks[j+5] * fator + crop_list[idx_][1])
            disp_landmark.append(display_landmark_x)
            disp_landmark.append(display_landmark_y)

        disp_landmarks.append(disp_landmark)

    for i in range(idx):
        for j in range(int(len(landmarks) / 2)):
            cv2.circle(img0, (int(disp_landmarks[i][j*2]), int(disp_landmarks[i][j*2+1])),  4, (0, 255, 0), -1)   # b g r
        cv2.rectangle(img0, (int(original_detect[i][0]), int(original_detect[i][1])),
                      (int(original_detect[i][2]), int(original_detect[i][3])), (0, 255, 0), 4)
        # (0, 255, 255) yellow

    cv2.imshow("landmarks_10_total", img0)
    cv2.waitKey(0)

    # cv2.imwrite("final_result.jpg", img0)


def convert_to_square(bbox):
    """
    convert bbox to square 将输入边框变为正方形，以最长边为基准，不改变中心点
    :param bbox: input bbox / numpy array , shape n x 5
    :return: square bbox
    """
    square_bbox = bbox.copy()   # bbox = [x1, y1, x2, y2]

    h = bbox[3] - bbox[1] + 1   # 计算框的宽度 加1？
    w = bbox[2] - bbox[0] + 1   # 计算框的长度
    max_side = np.maximum(h, w)       # 找出最大的那个边

    square_bbox[0] = bbox[0] + w*0.5 - max_side*0.5  # 新bbox的左上角x
    square_bbox[1] = bbox[1] + h*0.5 - max_side*0.5  # 新bbox的左上角y
    square_bbox[2] = square_bbox[0] + max_side - 1   # 新bbox的右下角x
    square_bbox[3] = square_bbox[1] + max_side - 1   # 新bbox的右下角y

    if square_bbox[0] < 0:
        lack_ = abs(square_bbox[0])
        square_bbox[0] = 0
        square_bbox[2] = square_bbox[2] + lack_

    if square_bbox[1] < 0:
        lack_ = abs(square_bbox[1])
        square_bbox[1] = 0
        square_bbox[3] = square_bbox[3] + lack_

    return square_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Test MTCNN Model(With Landmark)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['model/pnet', 'model/rnet', 'model/onet'], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[12, 15, 9], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.5, 0.5, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=40, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=-1, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    st = time.time()
    args = parse_args()
    print('Called with argument:')
    print(args)    # 输出程序执行中相关参数
    ctx = mx.gpu(args.gpu_id)  # 根据给定的gpu_id值,选择对应的GPU来存储和计算
    if args.gpu_id == -1:      # 如果gpu_id值为-1,那么我们采用CPU来存储和计算
        ctx = mx.cpu(0)
    crop, detect_len, orig_detect, idx, img, img_array = test_net(args.prefix, args.epoch, args.batch_size, ctx, args.thresh, args.min_face, args.stride,args.slide_window)
    test_landmark_net(crop, detect_len, orig_detect, idx, img, img_array)
    print('Session Done! Take {} s'.format(time.time() - st))



