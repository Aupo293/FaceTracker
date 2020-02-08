# -*- coding: UTF-8 -*-
import cv2
import time
from mtcnn_my import MTCNN
from facetrack import FaceTracking
import numpy as np
import os


def main_video():       # 读取视频版本
    video_path = '/Users/qiuxiaocong/Downloads/test_example.mp4'
    cap = cv2.VideoCapture(video_path)
    process = True
    frameindex = 0
    IDs = []
    Colors = []
    faceTrack = FaceTracking()
    idx = 0

    while process:
        ret, frame = cap.read()
        if not ret:   # 视频结束
            print('Last Frame!')
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        start = time.time()   # Counting Time

        if frameindex == 0:
            faceTrack.Init(frame)          # 首帧初始化
            if faceTrack.init_success():
                frameindex = 1
            else:
                continue
        else:                              # 非首帧处理
            faceTrack.update(frame)
        print("Total takes {}s\n".format(time.time()-start))  # Time
        # print("------------------\n")

        faceActions = faceTrack.trackingFace    # trackingFace存放跟踪的结果
        # print(len(faceActions))
        for i in range(len(faceActions)):       # 遍历所有的类别
            info = faceActions[i]
            x1, y1, x2, y2 = info.bbox  # 根据info获取 cv2.rectangle() 对角点坐标
            isExist = False
            for j in range(len(IDs)):
                if IDs[j] == info.face_id:
                    color = Colors[j]
                    isExist = True
                    break

            if not isExist:
                IDs.append(info.face_id)
                r = np.random.randint(255) + 1
                g = np.random.randint(255) + 1
                b = np.random.randint(255) + 1
                color = (r, g, b)
                Colors.append(color)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            for k in range(5):   # 标定5 points
                cv2.circle(frame, (int(info.face_5_points[k*2]), int(info.face_5_points[k*2+1])), 2, color, -1)  # (b,g,r) -1代表填充
            # print('get here')
        # print('here')
        # cv2.imshow("frame", frame)
        # cv2.imwrite(os.path.join('/DATA/disk1/qxc/facetrack_python/FaceTracker/result', '{}.jpg'.format(idx)), frame)

        # print(idx)
        idx = idx + 1
        # cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

def main_camera():            # 读取摄像头版本[更改本]
    cv2.namedWindow("Camera Preview")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():  # try to get the first frame
        print('Camera open fail.')

    #
    # width = 1280  # 640
    # height = 720  # 480
    # dim = (width, height)
    # # resize image
    # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # print('Resized Dimensions : ', resized.shape)

    process = True
    frameindex = 0
    IDs = []
    Colors = []
    faceTrack = FaceTracking()
    # idx = 0

    while process:
        rval, frame = cap.read()
        cv2.imshow("Resize Preview", cv2.flip(frame, 1))
        
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

        start = time.time()  # Counting Time

        if frameindex == 0:
            faceTrack.Init(frame)  # 首帧初始化
            frameindex = 1
        else:  # 非首帧处理
            faceTrack.update(frame)
        print("Total takes {}s\n".format(time.time() - start))  # Time
        # print("------------------\n")

        faceActions = faceTrack.trackingFace  # trackingFace存放跟踪的结果
        for i in range(len(faceActions)):  # 遍历所有的类别
            info = faceActions[i]
            x1, y1, x2, y2 = info.bbox  # 根据info获取 cv2.rectangle() 对角点坐标
            isExist = False
            for j in range(len(IDs)):
                if IDs[j] == info.face_id:
                    color = Colors[j]
                    isExist = True
                    break

            if not isExist:
                IDs.append(info.face_id)
                r = np.random.randint(255) + 1
                g = np.random.randint(255) + 1
                b = np.random.randint(255) + 1
                color = (r, g, b)
                Colors.append(color)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            for k in range(5):  # 标定5 points
                cv2.circle(frame, (int(info.face_5_points[k * 2]), int(info.face_5_points[k * 2 + 1])), 2, color,
                           -1)  # (b,g,r) -1代表填充

        # print('here')
        cv2.imshow("frame", frame)

        # print(idx)
        # idx = idx + 1

    cap.release()
    cv2.destroyWindow("Resize Preview")       #


# def main():            # 读取摄像头版本
#     cv2.namedWindow("Camera Preview")
#     vc = cv2.VideoCapture(0)
#
#     if vc.isOpened():  # try to get the first frame
#         rval, frame = vc.read()
#         print('Original Dimensions : ', frame.shape)
#     else:
#         rval = False
#     #
#     # width = 1280  # 640
#     # height = 720  # 480
#     # dim = (width, height)
#     # # resize image
#     # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     # print('Resized Dimensions : ', resized.shape)
#
#     frameindex = 0
#     IDs = []
#     Colors = []
#     faceTrack = FaceTracking()
#     idx = 0
#
#     while rval:
#         cv2.imshow("Resize Preview", cv2.flip(frame, 1))
#         rval, frame = vc.read()
#         # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#
#         start = time.time()  # Counting Time
#
#         if frameindex == 0:
#             faceTrack.Init(frame)  # 首帧初始化
#             frameindex = 1
#         else:  # 非首帧处理
#             faceTrack.update(frame)
#         print("Total takes {}s\n".format(time.time() - start))  # Time
#         # print("------------------\n")
#
#         faceActions = faceTrack.trackingFace  # trackingFace存放跟踪的结果
#         for i in range(len(faceActions)):  # 遍历所有的类别
#             info = faceActions[i]
#             x1, y1, x2, y2 = info.bbox  # 根据info获取 cv2.rectangle() 对角点坐标
#             isExist = False
#             for j in range(len(IDs)):
#                 if IDs[j] == info.face_id:
#                     color = Colors[j]
#                     isExist = True
#                     break
#
#             if not isExist:
#                 IDs.append(info.face_id)
#                 r = np.random.randint(255) + 1
#                 g = np.random.randint(255) + 1
#                 b = np.random.randint(255) + 1
#                 color = (r, g, b)
#                 Colors.append(color)
#
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#             for k in range(5):  # 标定5 points
#                 cv2.circle(frame, (int(info.face_5_points[k * 2]), int(info.face_5_points[k * 2 + 1])), 2, color,
#                            -1)  # (b,g,r) -1代表填充
#
#         # print('here')
#         cv2.imshow("frame", frame)
#
#         print(idx)
#         idx = idx + 1
#
#     vc.release()
#     cv2.destroyWindow("Resize Preview")       #


if __name__ == '__main__':
    main_video()

