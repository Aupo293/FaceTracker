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


def MTCNN(img):
    prefix = ['model/pnet', 'model/rnet', 'model/onet', 'model/lnet']
    epoch = [12, 15, 9, 4390]
    batch_size = [2048, 256, 16]
    thresh = [0.5, 0.6, 0.7]
    min_face_size = 40
    stride = 2
    slide_window = False
    ctx = mx.cpu()
    # ctx = mx.gpu()    
    # ctx = [mx.gpu(int(i)) for i in [0,1,2,3]]    

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
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

    mtcnn_detector = MtcnnDetector1(detectors=detectors, ctx=ctx, min_face_size=min_face_size, stride=stride, threshold=thresh, slide_window=slide_window)

    img_disp = img.copy()
    
    time1 = time.time()
    boxes, boxes_c = mtcnn_detector.detect_pnet(img)
    # print(boxes_c)
    time2 = time.time()
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    # print(boxes_c)
    time3 = time.time()
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)
    time4 = time.time()

    if boxes_c is not None:
        print('新检测到人脸!')
    else:
        print('此次检测未检测到人脸!')

    original_detect = []    # 存放经过onet得到的所有框[矩形框]
    crop_list = []          # 存放[矩形框]校正为[正方形框]后的结果
    detect_len_list = []
    nd_array = []           # 存放要送入到lnet的正方形框图像
    score_list = []         # 存放各个[矩形框]的inference分数
    idx = 0  # 人脸框计数索引

    def convert_to_square(bbox):
        """
        convert bbox to square 将输入边框变为正方形，以最长边为基准，不改变中心点
        :param bbox: input bbox / numpy array , shape n x 5
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

    if boxes_c is not None:
        # font = cv2.FONT_HERSHEY_SIMPLEX  # Python 一种字体

        for b in boxes_c:  # nms和bbr之后的结果
            # 在draw上绘制矩形框(左上角坐标+右下角坐标)
            b_new0 = np.array(b[0:4])  # 添加检测框
            original_detect.append(b_new0)
            """
            b_new = convert_to_square(b_new0)  # 添加送入到landmark net的48*48大小的框
            crop_list.append(b_new)
            score_list.append(b[4])

            # img_draw为校正为[正方形框]之后的四个角坐标值所框定的图片区域
            img_draw = img[int(b_new[1]):int(b_new[3]), int(b_new[0]):int(b_new[2])]
            detect_len = min(img_draw.shape[0], img_draw.shape[1])
            if detect_len != 0:
                detect_len_list.append(detect_len)

                img_resized = cv2.resize(img_draw, (48, 48))
                nd_array.append(img_resized)
                idx = idx + 1
            """
    time5 = time.time()

    """
    # load lnet model(1990 3330 4390)
    sym = L_Net('test')
    args, auxs = load_param(prefix[3], epoch[3], convert=False, ctx=ctx)

    data_size = 48      # landmark net 输入的图像尺寸为48*48
    imshow_size = 48    # imshow_size为landmark结果展示的图片尺寸

    data_shapes = {'data': (1, 3, data_size, data_size)}
    disp_landmarks = []

    for idx_ in range(idx):
        img_lnet = nd_array[idx_]
        newimg = transform(img_lnet)
        args['data'] = mx.nd.array(newimg, ctx)
        executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))
        executor.copy_params_from(args, auxs)

        out_list = [[] for _ in range(len(executor.outputs))]
        executor.forward(is_train=False)   # inference
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

        landmarks = landmarks * imshow_size  # landmarks输出值应该在0~1 需复原
        landmarks = np.reshape(landmarks, -1)

        fator = detect_len_list[idx_]/48.0
        disp_landmark = []

        for j in range(int(len(landmarks) / 2)):
            display_landmark_x = int(landmarks[j] * fator + crop_list[idx_][0])
            display_landmark_y = int(landmarks[j+5] * fator + crop_list[idx_][1])
            disp_landmark.append(display_landmark_x)
            disp_landmark.append(display_landmark_y)

        disp_landmarks.append(disp_landmark)
    """
    time6 = time.time()

    # result = dict()
    # for i in range(idx):
    #     result[i] = [disp_landmarks[i], original_detect[i], score_list[i]]
    # print(result)

    # for i in range(idx):
    #     for j in range(int(len(landmarks) / 2)):
    #         cv2.circle(img_disp, (int(disp_landmarks[i][j*2]), int(disp_landmarks[i][j*2+1])),  2, (0, 255, 0), -1)   # b g r
    #     cv2.rectangle(img_disp, (int(original_detect[i][0]), int(original_detect[i][1])), (int(original_detect[i][2]), int(original_detect[i][3])), (0, 255, 0), 2)  # (0, 255, 255) yellow

    # cv2.imshow("landmarks_10_total", img_disp)
    # cv2.waitKey(0)
    # cv2.imwrite("final_result.jpg", img0)

    # time5 = time.time()
    print('time2-time1:{}'.format(time2-time1))
    print('time3-time2:{}'.format(time3-time2))
    print('time4-time3:{}'.format(time4-time3))
    print('time5-time4:{}'.format(time5-time4))
    print('time6-time5:{}'.format(time6 - time5))

    # return result
    return original_detect




