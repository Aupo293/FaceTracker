from mtcnn_my import MTCNN
import numpy as np
import cv2
import os
# from mtcnn import MTCNN as mt

def convert_to_square(bbox):
    """
    convert bbox to square 将输入边框变为正方形，以最长边为基准，不改变中心点
    :param bbox: input bbox / numpy array , shape n x 5 [4个坐标值]
    :return: square bbox
    """
    square_bbox = bbox.copy()  # bbox = [x1, y1, x2, y2]

    h = bbox[3] - bbox[1] + 1  # 计算框的宽度 加1？
    w = bbox[2] - bbox[0] + 1  # 计算框的长度
    max_side = np.maximum(h, w)  # 找出最大的那个边

    square_bbox[0] = bbox[0] + w * 0.5 - max_side * 0.5  # 新bbox的左上角x
    square_bbox[1] = bbox[1] + h * 0.5 - max_side * 0.5  # 新bbox的左上角y
    square_bbox[2] = square_bbox[0] + max_side - 1  # 新bbox的右下角x
    square_bbox[3] = square_bbox[1] + max_side - 1  # 新bbox的右下角y

    if square_bbox[0] < 0:
        lack_ = abs(square_bbox[0])
        square_bbox[0] = 0
        square_bbox[2] = square_bbox[2] + lack_

    if square_bbox[1] < 0:
        lack_ = abs(square_bbox[1])
        square_bbox[1] = 0
        square_bbox[3] = square_bbox[3] + lack_

    return square_bbox


# result = MTCNN('/Users/qiuxiaocong/Downloads/facetrack_python/cba.jpeg')
#
# print(result[0][1])
# print(type(result[0][1]))
#
# res = convert_to_square(result[0][1])
# print(res)
# print(res[0],res[1],res[2],res[3])
# print(type(res))

# [613, 136, 640, 132, 625, 152, 619, 167, 642, 164]
# <class 'list'>
# [599.41253397  97.09813888 669.6237403  186.66637458]
# <class 'numpy.ndarray'>
# 1.0
# <class 'numpy.float64'>


# a = np.random.randint(255)
# print(a)

#
# class Tset(object):
#     def __init__(self):
#         self.image = None
#
#     def add(self, x):
#         x = x + 1
#         return x
#
#     def test(self):
#         self.image = 10
#         for i in range(5):
#             self.image = self.add(self.image)
#         print(self.image)
#
#
# a = Tset()
# a.test()


# img = '/Users/qiuxiaocong/Downloads/facetrack_python/cba.jpeg'
# image = cv2.imread(img)
# image
# image[int(180):int(459), int(346):int(549)] = 0
# cv2.imshow('frame', image)
# cv2.waitKey(0)
#
# a = list(filter(lambda x: x %2 ==0, [0,1,2,3,4,5,6,7,8,9]))
# print(a)


#
# img = '/Users/qiuxiaocong/Downloads/facetrack_python/cba.jpeg'
# image = cv2.imread(img)
# print(image.shape)
# w,h = image.shape[::-1]
# print(w, h)
# cv2.imshow('a', image)
# cv2.waitKey(0)


# def tracking_corrfilter(frame, model):
#     frame_disp = frame.copy()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     model_gray = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
#     # x1, y1, x2, y2 = trackBox[0], trackBox[1], trackBox[2], trackBox[3]
#     # w = x2 - x1 + 1
#     # h = y2 - y1 + 1
#     # print(model_gray.shape[::-1])
#     w, h = model_gray.shape[::-1]
#
#     # Apply template Matching
#     method = eval('cv2.TM_CCOEFF_NORMED')
#     res = cv2.matchTemplate(frame_gray, model_gray, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     top_left = max_loc                                  # 左上角
#     bottom_right = (top_left[0] + w, top_left[1] + h)   # 右下角
#     trackBox = np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
#     # cv2.rectangle(frame_disp, top_left, bottom_right, 255, 2)
#     # cv2.imshow('test', frame_disp)
#     # cv2.waitKey(0)
#     return trackBox
#
# frame = cv2.imread('/Users/qiuxiaocong/Downloads/facetrack_python/cba.jpeg')
# model = cv2.imread('/Users/qiuxiaocong/Desktop/model.png')
#
# tracking_corrfilter(frame, model)





# -*- coding: UTF-8 -*-
import cv2
import time
from mtcnn_my import MTCNN
import numpy as np
import time
from core.symbol import L_Net
from tools.load_model import load_param
import mxnet as mx
from tools.image_processing import transform


# -*- coding: UTF-8 -*-
import numpy as np
import mxnet as mx
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
import argparse
import cv2
import time
from core import *

# sym = O_Net('test')
# ctx = mx.cpu()
# args, auxs = load_param('model/onet', 9, convert=False, ctx=ctx)
#
# img = cv2.imread('/Users/qiuxiaocong/Downloads/facetrack_python/detection_result.jpg')
# print(img.shape)

# data_size = 48  # landmark net 输入的图像尺寸为48*48
# imshow_size = 48  # imshow_size为landmark结果展示的图片尺寸
# data_shapes = {'data': (1, 3, data_size, data_size)}
# #
# # # img_resized = cv2.resize(image, (48, 48))
# #
# #
# newimg = transform(img)
# args['data'] = mx.nd.array(newimg, ctx)
# executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))
# executor.copy_params_from(args, auxs)
# executor.forward(is_train=False)  # inference
# #
# # print(executor.outputs)
# # time.sleep(1000000)
#
# out_list = [[] for _ in range(len(executor.outputs))]
# # executor.forward(is_train=False)  # inference
# for o_list, o_nd in zip(out_list, executor.outputs):
#     o_list.append(o_nd.asnumpy())
# out = list()
# for o in out_list:
#     out.append(np.vstack(o))
#
# cls_pro = out[0][0][1]
# print(cls_pro)
#
# time.sleep(1000000)
#
# img = cv2.imread('/Users/qiuxiaocong/Downloads/facetrack_python/error.jpg')
# print(img.shape)
#
# a = mt()
# result = a.detect_faces(img)
#
# print(result)
# cv2.imshow('aaa', img)
# cv2.waitKey(0)
# video_path = '/Users/qiuxiaocong/Downloads/test_example.mp4'
# cap = cv2.VideoCapture(video_path)
# idx = 0
# while True:
#     res, frame = cap.read()
#     if not res:
#         print('Not Frame')
#         break
#     cv2.imwrite(os.path.join('/Users/qiuxiaocong/Downloads/test_example', '{}.jpg'.format(idx)), frame)
#     idx = idx + 1
# cap.release()
# cv2.destroyAllWindows()


# [{'box': [13, -10, 74, 98], 'confidence': 0.783650815486908, 'keypoints': {'left_eye': (33, 21), 'right_eye': (68, 18), 'nose': (56, 43), 'mouth_left': (40, 63), 'mouth_right': (68, 61)}}]


# def calibrate_box(bbox, reg):
#     """
#     calibrate bboxes 校准BBox(Bounding Box Regression)
#     :param bbox: input bboxes / numpy array, shape n x 5
#     :param reg: bboxes adjustment / numpy array, shape n x 4
#     :return: bboxes after refinement
#     """
#     bbox_c = bbox.copy()
#     w = bbox[2] - bbox[0] + 1  # 计算框的长度 加1？
#     # w = np.expand_dims(w, 1)
#     h = bbox[3] - bbox[1] + 1  # 计算框的宽度
#     # h = np.expand_dims(h, 1)
#     reg_m = np.hstack([w, h, w, h])  # 在水平方向上平铺
#     aug = reg_m * reg  # aug应该是回归量
#     bbox_c[0:4] = bbox_c[0:4] + aug
#     return bbox_c  # 返回校正之后的bbox_c


# def onet_detector(image):
#     """
#     :param image: 输入为48*48大小的图像
#     :return:    返回概率值
#     """
#     sym = O_Net('test')
#     ctx = mx.cpu()
#     args, auxs = load_param('model/onet', 9, convert=False, ctx=ctx)
#     data_size = 48  # landmark net 输入的图像尺寸为48*48
#     data_shapes = {'data': (1, 3, data_size, data_size)}
#     # # img_resized = cv2.resize(image, (48, 48))
#
#     newimg = transform(image)
#     args['data'] = mx.nd.array(newimg, ctx)
#     executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))
#     executor.copy_params_from(args, auxs)
#     executor.forward(is_train=False)  # inference
#     out_list = [[] for _ in range(len(executor.outputs))]
#     for o_list, o_nd in zip(out_list, executor.outputs):
#         o_list.append(o_nd.asnumpy())
#     out = list()
#     for o in out_list:
#         out.append(np.vstack(o))
#     cls_pro = out[0][0][1]
#     return out[1][0]
#
# img = cv2.imread('/Users/qiuxiaocong/Downloads/facetrack_python/detection_result.jpg')
# print(img.shape)
# result = onet_detector(img)
# print(result)
# new = calibrate_box(np.array([0,0,48,48]), result)
# print(new)
# cv2.rectangle(img, (new[0],new[1]),(new[2],new[3]),255,2)
# cv2.imshow('a',img)
# cv2.waitKey(0)



# cap = cv2.VideoCapture(0)
# cv2.namedWindow("Resize Preview")
# time.sleep(2)
# while True:
#     res, frame = cap.read()
#     cv2.imshow('now', frame)



import cv2
#!/usr/bin/env python3
# import cv2
# cv2.namedWindow("Resize Preview")
# vc = cv2.VideoCapture(0)
#
# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
#     print('Original Dimensions : ',frame.shape)
# else:
#     rval = False
#
# width = 640
# height = 480
# dim = (width, height)
# # resize image
# resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
# print('Resized Dimensions : ',resized.shape)
#
# while rval:
#     cv2.imshow("Resize Preview", cv2.flip(frame, 1))
#     rval, frame = vc.read()
#     frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#         break
# cv2.destroyWindow("Resize Preview")


# a = {}
# for i in range(len(a)):
#     print('heool')





