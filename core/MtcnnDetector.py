import cv2
import mxnet as mx
import time
from tools import image_processing
#from mx.model import FeedForward
import numpy as np
from config import config
from tools.nms import py_nms


class MtcnnDetector1(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a mxnet version
    """
    def __init__(self,
                 detectors,
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,   # 图像金字塔缩放参数
                 ctx=mx.cpu(),
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.ctx = ctx
        self.scale_factor = scale_factor
        self.slide_window = slide_window  # P_Net是否采用滑动窗口来提取Region Proposal,我们采用的是FCN全卷积网络来实现

    def convert_to_square(self, bbox):
        """
        convert bbox to square 将输入边框变为正方形，以最长边为基准，不改变中心点
        :param bbox: input bbox / numpy array , shape n x 5
        :return: square bbox
        """
        square_bbox = bbox.copy()   # bbox = [x1, y1, x2, y2]

        h = bbox[:, 3] - bbox[:, 1] + 1   # 计算框的宽度 加1？
        w = bbox[:, 2] - bbox[:, 0] + 1   # 计算框的长度
        max_side = np.maximum(h, w)       # 找出最大的那个边

        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5  # 新bbox的左上角
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5  # 新bbox的左上角
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1   # 新bbox的右下角
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1   # 新bbox的右下角
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
        calibrate bboxes 校准BBox(Bounding Box Regression)
        :param bbox: input bboxes / numpy array, shape n x 5
        :param reg: bboxes adjustment / numpy array, shape n x 4
        :return: bboxes after refinement
        """
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1   # 计算框的长度 加1？
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1   # 计算框的宽度
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])   # 在水平方向上平铺
        aug = reg_m * reg                 # aug应该是回归量
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c    # 返回校正之后的bbox_c

    def generate_bbox(self, map, reg, scale, threshold):
        """
        generate bbox from cls_map according to the threshold  从特征图中生成Bounding Box（将P/R/O_NET得到的region Proposal 映射回原图上的初始位置）
        :param map: detect score for each position / numpy array , n x m x 1
        :param reg: bbox / numpy array , n x m x 4
        :param scale: scale of this detection / float number
        :param threshold: detect threshold / float number
        :return: bbox array
        """
        # Bounding Box array:[x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset]
        # x和y都是根据特征图反采样得来的坐标，表示特征图这个点表达了原图哪个地方的特征
        # x1_offset and y1_offset are the prediction of PNet.
        # x1_offset和y1_offset都是PNet的预测结果。

        stride = 2      # 步长为2
        cellsize = 12   # 每个(滑动窗口?)的尺寸为:12×12

        t_index = np.where(map > threshold)
        # 这里的map是np.array型的,np.where返回的是map中所有满足条件的索引值。
        # 概率值map大于阈值，我们就认为这个Bounding Box是符合要求的。
        # t_index 中的每一个值应该包括 [[],[],[]]

        # find nothing 如果没有这样的 Bounding Box,则返回一个空的 np.array([])
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride*t_index[1])/scale),
                                 np.round((stride*t_index[0])/scale),
                                 np.round((stride*t_index[1]+cellsize)/scale),
                                 np.round((stride*t_index[0]+cellsize)/scale),
                                 score,
                                 reg])
        # scale：因为原图乘以scale进行缩放，所以缩放后图上点的左边应该按照scale放大回去，因此是除以 这张图的所对应原图的缩放因子。
        # 想要得到在原图上的坐标位置

        return boundingbox.T

    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]    这里应该是有点类似于图像金字塔的操作，长宽各自乘以scale得到不同尺寸的图片金字塔？
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image 采用最近邻插值
        img_resized = image_processing.transform(img_resized)
        return img_resized  # (batch_size, c, h, w)

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1   # bbox的宽和高
        num_box = bboxes.shape[0]   # bbox数量

        dx , dy= np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = self.resize_image(im, current_scale)
        _, _, current_height, current_width = im_resized.shape

        if self.slide_window:
            # sliding window   用滑动窗口来提取
            temp_rectangles = list()
            rectangles = list()     # list of rectangles [x11, y11, x12, y12, confidence] (corresponding to original image)
            all_cropped_ims = list()
            while min(current_height, current_width) > net_size:
                current_y_list = range(0, current_height - net_size + 1, self.stride) if (current_height - net_size) % self.stride == 0 \
                else range(0, current_height - net_size + 1, self.stride) + [current_height - net_size]
                current_x_list = range(0, current_width - net_size + 1, self.stride) if (current_width - net_size) % self.stride == 0 \
                else range(0, current_width - net_size + 1, self.stride) + [current_width - net_size]

                for current_y in current_y_list:
                    for current_x in current_x_list:
                        cropped_im = im_resized[:, :, current_y:current_y + net_size, current_x:current_x + net_size]

                        current_rectangle = [int(w * float(current_x) / current_width), int(h * float(current_y) / current_height),
                                             int(w * float(current_x) / current_width) + int(w * float(net_size) / current_width),
                                             int(h * float(current_y) / current_height) + int(w * float(net_size) / current_width),
                                                 0.0]
                        temp_rectangles.append(current_rectangle)
                        all_cropped_ims.append(cropped_im)

                current_scale *= self.scale_factor
                im_resized = self.resize_image(im, current_scale)
                _, _, current_height, current_width = im_resized.shape

            '''
            # helper for setting PNet batch size
            num_boxes = len(all_cropped_ims)
            batch_size = self.pnet_detector.batch_size
            ratio = float(num_boxes) / batch_size
            if ratio > 3 or ratio < 0.3:
                print "You may need to reset PNet batch size if this info appears frequently, \
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
            '''
            all_cropped_ims = np.vstack(all_cropped_ims)
            cls_scores, reg = self.pnet_detector.predict(all_cropped_ims)

            cls_scores = cls_scores[:, 1].flatten()
            keep_inds = np.where(cls_scores > self.thresh[0])[0]

            if len(keep_inds) > 0:
                boxes = np.vstack(temp_rectangles[ind] for ind in keep_inds)
                boxes[:, 4] = cls_scores[keep_inds]
                reg = reg[keep_inds].reshape(-1, 4)
            else:
                return None, None


            keep = py_nms(boxes, 0.7, 'Union')
            boxes = boxes[keep]

            boxes_c = self.calibrate_box(boxes, reg[keep])

        else:
            # fcn
            all_boxes = list()
            while min(current_height, current_width) > net_size:  # 图片金字塔
                cls_map, reg = self.pnet_detector.predict(im_resized)
                cls_map = cls_map.asnumpy()
                reg = reg.asnumpy()
                # 根据阈值生成框
                boxes = self.generate_bbox(cls_map[0, 1, :, :], reg, current_scale, self.thresh[0])

                current_scale *= self.scale_factor
                im_resized = self.resize_image(im, current_scale)
                _, _, current_height, current_width = im_resized.shape

                if boxes.size == 0:
                    continue
                keep = py_nms(boxes[:, :5], 0.5, 'Union')
                boxes = boxes[keep]
                all_boxes.append(boxes)

            if len(all_boxes) == 0:
                return None, None

            all_boxes = np.vstack(all_boxes)

            # merge the detection from first stage
            keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
            all_boxes = all_boxes[keep]
            boxes = all_boxes[:, :5]

            bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
            bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

            # refine the boxes
            boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                                 all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                                 all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                                 all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                                 all_boxes[:, 4]])
            # all_boxes[:,4]对应这个box的分数score
            boxes_c = boxes_c.T

        return boxes, boxes_c

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = image_processing.transform(cv2.resize(tmp, (24, 24)))

        cls_scores, reg = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1].flatten()
        keep_inds = np.where(cls_scores > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = py_nms(boxes, 0.7)
        boxes = boxes[keep]

        boxes_c = self.calibrate_box(boxes, reg[keep])
        # 先NMS去除掉多余的region Proposal，再进行Bounding box regress
        return boxes, boxes_c

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting ONet batch size
        batch_size = self.onet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset ONet batch size if this info appears frequently, \
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = image_processing.transform(cv2.resize(tmp, (48, 48)))
        cls_scores, reg = self.onet_detector.predict(cropped_ims)

        cls_scores = cls_scores[:, 1].flatten()
        keep_inds = np.where(cls_scores > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        boxes_c = self.calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, 0.7, "Minimum")
        boxes_c = boxes_c[keep]

        return boxes, boxes_c

    def detect_face(self, imdb, test_data, vis):
        """Detect face over image

        Parameters:
        ----------
        imdb: imdb
            image database
        test_data: data iter
            test data iterator
        vis: bool
            whether to visualize detection results

        Returns: 经过O_NET之后最终得到的Bounding Box
        -------
        """
        all_boxes = list()
        batch_idx = 0
        for databatch in test_data:
            if batch_idx % 100 == 0:
                print("%d images done"%batch_idx)
            im = databatch.data[0].asnumpy().astype(np.uint8)
            t = time.time()

            # pnet
            if self.pnet_detector:
                boxes, boxes_c = self.detect_pnet(im)
                # print(boxes_c)
                if boxes_c is None:    # 框回归校正后为None?
                    all_boxes.append(np.array([]))
                    batch_idx += 1
                    continue
                if vis:   # 是否可视化
                    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    self.vis_two(rgb_im, boxes, boxes_c)

                t1 = time.time() - t
                t = time.time()

            # rnet
            if self.rnet_detector:
                boxes, boxes_c = self.detect_rnet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    batch_idx += 1
                    continue
                if vis:
                    self.vis_two(rgb_im, boxes, boxes_c)

                t2 = time.time() - t
                t = time.time()

            # onet
            if self.onet_detector:
                boxes, boxes_c = self.detect_onet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    batch_idx += 1
                    continue
#                all_boxes.append(boxes_c)
                if vis:
                    self.vis_two(rgb_im, boxes, boxes_c)

                t3 = time.time() - t
                t = time.time()
                # print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

            all_boxes.append(boxes_c)
            batch_idx += 1
        # save detections into fddb format
#        imdb.write_results(all_boxes)
        return all_boxes    # 得到test_data中所有图片经过最终O_NET输出的校正后的Bounding Box

    def vis_two(self, im_array, dets1, dets2, thresh=0.9):
        """Visualize detection results before and after calibration
        可视化 calibration 前后的Bounding Box 高于阈值和低于阈值用不同颜色框表示
        Parameters:
        ----------
        im_array: numpy.ndarray, shape(1, c, h, w)
            test image in rgb
        dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
            detection results before calibration
        dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
            detection results after calibration
        thresh: float
            boxes with scores > thresh will be drawn in red otherwise yellow

        Returns:
        -------
        """
        import matplotlib.pyplot as plt
        import random

        figure = plt.figure()
        plt.subplot(121)
        plt.imshow(im_array)
        color = 'yellow'

        for i in range(dets1.shape[0]):
            bbox = dets1[i, :4]
            score = dets1[i, 4]
            if score > thresh:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor='red', linewidth=0.7)
                plt.gca().add_patch(rect)
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:.3f}'.format(score),
                               bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
            else:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=0.5)
                plt.gca().add_patch(rect)

        plt.subplot(122)
        plt.imshow(im_array)
        color = 'yellow'

        for i in range(dets2.shape[0]):
            bbox = dets2[i, :4]
            score = dets2[i, 4]
            if score > thresh:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor='red', linewidth=0.7)
                plt.gca().add_patch(rect)
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:.3f}'.format(score),
                               bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
            else:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=0.5)
                plt.gca().add_patch(rect)
        plt.show()
