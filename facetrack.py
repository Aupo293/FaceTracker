# -*- coding: UTF-8 -*-
import cv2
import time
from mtcnn_my import MTCNN
import numpy as np
import time
from core.symbol import L_Net, O_Net
from tools.load_model import load_param
import mxnet as mx
from tools.image_processing import transform
from multiprocessing import Pool
from functools import partial


class Face(object):
    def __init__(self):
        self.face_id = None
        self.loc = None
        self.isCanShow = None
        self.frame_face_prev = None
        self.face_5_points = []
        self.frameId = 0
        self.ptr_num = 0
        self.score = None
        self.bbox = None

    def face(self, instance_id, rect):
        self.face_id = instance_id
        self.loc = rect
        self.isCanShow = False


class FaceTracking(object):
    def __init__(self):
        self.tracking_id = None
        self.ImageHighDP = None
        self.candidateFaces_lock = None
        self.detection_Time = -1
        self.detection_Interval = None
        self.stabilization = None
        self.trackingFace = []      # 输出的结果
        self.candidateFaces = []
        self.tpm_scale = 2
        # self.face = Face()

    def detecting(self, image):
        try:
            mtcnn_result = MTCNN(image)   # MTCNN人脸检测结果 首帧人脸检测
        except:
            mtcnn_result = {}
        box_num = len(mtcnn_result)   # 首帧包含的人脸个数
        self.candidateFaces_lock = 1  # 上锁
        for i in range(box_num):
            result = mtcnn_result[i]  # result 包含[[5 points],array[box],score]
            bbox = result[1]          # 0 -> 5 points, 1 -> bbox, 2 ->score
            bbox_square = self.convert_to_square(bbox)  # 校正为正方形  [x1, y1, x2, y2]

            face_new = Face()
            face_new.face(self.tracking_id, bbox_square)

            img_draw = image[int(bbox_square[1]):int(bbox_square[3]), int(bbox_square[0]):int(bbox_square[2])]
            face_new.frame_face_prev = img_draw

            self.tracking_id = self.tracking_id + 1   # 统计待追踪的个数
            self.candidateFaces.append(face_new)      # self.candidateFaces 存放的是类!

        self.candidateFaces_lock = 0

    def Init(self, image):
        # self.ImageHighDP = image.copy()   # 复制输入图片
        self.tracking_id = 0              # 初始化追踪id为0
        self.detection_Interval = 0.5     # 检测间隔 detect faces every 200 ms
        self.detecting(image)             # 首帧人脸检测
        self.stabilization = False

    @staticmethod
    def tracking_corrfilter(frame, model):
        frame_disp = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        model_gray = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
        # x1, y1, x2, y2 = trackBox[0], trackBox[1], trackBox[2], trackBox[3]
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        w, h = model_gray.shape[::-1]

        # Apply template Matching  [eval() 函数用来执行一个字符串表达式，并返回表达式的值]
        # https://segmentfault.com/a/1190000015679691
        # http://bluewhale.cc/2017-09-22/use-python-opencv-for-image-template-matching-match-template.html
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(frame_gray, model_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc                                  # 左上角
        bottom_right = (top_left[0] + w, top_left[1] + h)   # 右下角

        # cv2.rectangle(frame_disp, top_left, bottom_right, 255, 2)   # display result
        # cv2.imshow('test', frame_disp)
        # cv2.waitKey(0)

        trackBox = np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
        return trackBox

    @staticmethod
    def onet_detector(image):
        """
        :param image: 输入为48*48大小的图像
        :return:    返回概率值
        """
        sym = O_Net('test')
        ctx = mx.cpu()
        # ctx = mx.gpu()       
        # ctx = [mx.gpu(int(i)) for i in [0,1,2,3]]

        args, auxs = load_param('model/onet', 9, convert=False, ctx=ctx)
        data_size = 48  # landmark net 输入的图像尺寸为48*48
        data_shapes = {'data': (1, 3, data_size, data_size)}
        # # img_resized = cv2.resize(image, (48, 48))

        newimg = transform(image)
        args['data'] = mx.nd.array(newimg, ctx)
        executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))
        executor.copy_params_from(args, auxs)
        executor.forward(is_train=False)  # inference
        out_list = [[] for _ in range(len(executor.outputs))]
        for o_list, o_nd in zip(out_list, executor.outputs):
            o_list.append(o_nd.asnumpy())
        out = list()
        for o in out_list:
            out.append(np.vstack(o))
        cls_pro = out[0][0][1]
        return out

    @staticmethod
    def calibrate_box(bbox, reg):
        """
        calibrate bboxes 校准BBox(Bounding Box Regression)
        :param bbox: input bboxes / numpy array, shape n x 5
        :param reg: bboxes adjustment / numpy array, shape n x 4
        :return: bboxes after refinement
        """
        bbox_c = bbox.copy()
        w = bbox[2] - bbox[0] + 1   # 计算框的长度 加1？
        # w = np.expand_dims(w, 1)
        h = bbox[3] - bbox[1] + 1   # 计算框的宽度
        # h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])   # 在水平方向上平铺
        aug = reg_m * reg                 # aug应该是回归量
        bbox_c[0:4] = bbox_c[0:4] + aug
        return bbox_c    # 返回校正之后的bbox_c

    def doingLandmark_onet(self, image, trackBox):
        """

        :param image:
        :param trackBox:
        :return:
        """
        # x1 = trackBox[0]
        # y1 = trackBox[1]
        #
        # cv2.imwrite('error.jpg', image)
        # mtcnn_result = MTCNN(image)
        # print(mtcnn_result)
        # cls_pro = mtcnn_result[0][2]  # 0 -> 5 points, 1 -> bbox, 2 ->score
        # bbox = mtcnn_result[0][1]
        # bbox[0] = bbox[0] + x1
        # bbox[1] = bbox[1] + y1
        # bbox[2] = bbox[2] + x1
        # bbox[3] = bbox[3] + y1
        # landmarks = mtcnn_result[0][0]
        # landmarks[0] = landmarks[0] + x1
        # landmarks[1] = landmarks[1] + y1
        # landmarks[2] = landmarks[2] + x1
        # landmarks[3] = landmarks[3] + y1
        # landmarks[4] = landmarks[4] + x1
        # landmarks[5] = landmarks[5] + y1
        # landmarks[6] = landmarks[6] + x1
        # landmarks[7] = landmarks[7] + y1
        # landmarks[8] = landmarks[8] + x1
        # landmarks[9] = landmarks[9] + y1

        # bbox = list(bbox)

        # return cls_pro, bbox, landmarks

        detect_length = min(image.shape[0], image.shape[1])
        ctx = mx.cpu()
        # ctx = mx.gpu()        
        # ctx = [mx.gpu(int(i)) for i in [0,1,2,3]]

        sym = L_Net('test')
        args, auxs = load_param('model/lnet', 4390, convert=False, ctx=ctx)

        data_size = 48  # landmark net 输入的图像尺寸为48*48
        imshow_size = 48  # imshow_size为landmark结果展示的图片尺寸
        data_shapes = {'data': (1, 3, data_size, data_size)}

        img_resized = cv2.resize(image, (48, 48))
        result = self.onet_detector(img_resized)   # 得到该图是人脸的概率值
        cls_pro = result[0][0][1]
        reg_m = result[1][0]
        bbox_new = self.calibrate_box(trackBox, reg_m)

        newimg = transform(img_resized)
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

        fator = float(detect_length)/48.0
        disp_landmark = []

        for j in range(int(len(landmarks) / 2)):
            display_landmark_x = int(landmarks[j] * fator + trackBox[0])
            display_landmark_y = int(landmarks[j+5] * fator + trackBox[1])
            disp_landmark.append(display_landmark_x)
            disp_landmark.append(display_landmark_y)

        # for j in range(int(len(landmarks) / 2)):
        #     cv2.circle(frame, (int(disp_landmark[j * 2]), int(disp_landmark[j * 2 + 1])), 2, (0, 255, 0), -1)  # b g r
        # cv2.rectangle(frame, (int(trackBox[0]), int(trackBox[1])), (int(trackBox[2]), int(trackBox[3])), (0, 255, 0), 2)  #
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

        return cls_pro, bbox_new, disp_landmark

    def tracking(self, image, face):
        # face 为首帧得到的其中一个候选人脸对应的类Face, image可以认为是第二帧的image
        st = time.time()
        faceROI = face.loc    # 对应的是坐标
        model = face.frame_face_prev
        trackBox = self.tracking_corrfilter(image, model)
        time5 = time.time()

        trackBox_new = self.convert_to_square(trackBox)   # 转变为正方形 以便后续操作
        x1 = trackBox_new[0]
        y1 = trackBox_new[1]
        x2 = trackBox_new[2]
        y2 = trackBox_new[3]
        # x1, y1, x2, y2 = trackBox[0], trackBox[1], trackBox[2], trackBox[3]
        # 根据搜寻到的坐标 从当前帧获取对应图像区域即为faceROI_Image
        faceROI_Image = image[int(y1):int(y2), int(x1):int(x2)]   # 正方形图片

        time6 = time.time()

        # faceROI_Image为输入LNet的图像, face对应类 利用face.face_5_points存放5点关键点
        face.score, face.bbox, face.face_5_points = self.doingLandmark_onet(faceROI_Image, trackBox_new)  # ?

        time7 = time.time()

        if face.score > 0.1:
            face.loc = self.convert_to_square(face.bbox)
            img_draw = image[int(face.loc[1]):int(face.loc[3]), int(face.loc[0]):int(face.loc[2])]
            face.frame_face_prev = img_draw
            face.frameId += 1
            face.isCanShow = True
            
            time8 = time.time()
            
            print('time5-st1:{}'.format(time5-st))
            print('time6-time5:{}'.format(time6-time5))
            print('time7-time6:{}'.format(time7-time6))
            print('time8-time7:{}'.format(time8-time7))

            return True
        else:
            print('time5-st1:{}'.format(time5-st))
            print('time6-time5:{}'.format(time6-time5))
            print('time7-time6:{}'.format(time7-time6))

            return False

    def setMask(self, image, loc):   # 将image中的face区域置为0 face:x1 y1 x2 y2
        x1, y1, x2, y2 = loc[0], loc[1], loc[2], loc[3]
        image[int(y1):int(y2), int(x1):int(x2)] = 0
        return image

    def update(self, image):
        st = time.time()
        
        self.ImageHighDP = image.copy()   # 复制

        time1 = time.time()

        if len(self.candidateFaces) > 0 and not self.candidateFaces_lock:  # 同时检测完成
            for i in range(len(self.candidateFaces)):
                self.trackingFace.append(self.candidateFaces[i])
            self.candidateFaces.clear()

        time2 = time.time()

        # self.trackingFace中存放的是一个个的类Face
        # for i in range(len(self.trackingFace)):
        #     if not self.tracking(image, self.trackingFace[i]):
        # 不可以采用这种方法在for循环中删除元素 https://segmentfault.com/a/1190000007214571

        # self.trackingFace = list(filter(lambda x: self.tracking(image, x), self.trackingFace))
        # print(self.trackingFace)

        pool = Pool(processes=4)
        func = partial(self.tracking, image)
        result = pool.map(func, self.trackingFace[:])
        temp = []
        for i in range(len(result)):
            if result[i]:
                temp.append(self.trackingFace[i])
        self.trackingFace = temp

                
        time3 = time.time()

        if self.detection_Time < 0:
            self.detection_Time = time.time()
        else:
            diff = time.time() - self.detection_Time
            if diff > self.detection_Interval:
                for class_ in self.trackingFace:
                    self.ImageHighDP = self.setMask(self.ImageHighDP, class_.loc)
                self.detection_Time = time.time()
                # print('Have detected.')
                self.detecting(self.ImageHighDP)
        time4 = time.time()
        
        print('time1-st:{}'.format(time1 - st))
        print('time2-time1:{}'.format(time2 - time1)) 
        print('time3-time2:{}'.format(time3 - time2))
        print('time4-time3:{}'.format(time4 - time3))  


    @staticmethod
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


