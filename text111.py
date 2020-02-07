# coding:utf-8
import numpy as np
from scipy.io import loadmat, savemat
import os.path as op
import h5py
import pandas as pd
import time
import mxnet as mx
import os


# a = np.array([5,9,13,14,35,46,7])
# # b = 4
# throw1 = 10
# # throw2 = 6
# result1 = np.where(a>throw1)
# # result2 = np.where(b>throw2)
# print(result1)  # 大于的话是属于0？
# # print(result2)
# if result1[0].size == 0:
#     print(np.array([]))
# else:
#     print(result1[0].size)


# a = np.array([[[1],[2],[3]],[[4],[5],[6]]])
# print(a.shape)
# index_ = np.where(a>3)
# print(index_)
# print(index_[0])
# print(index_[1])

# data_file = '/Users/qiuxiaocong/Downloads/mtcnn-master/prepare_data/wider_annotations/wider_face_val.mat'
# data = h5py.File(data_file,'r')
# event_list = data['event_list/file_list/']
# print(event_list)

# anno_file = "/Users/qiuxiaocong/Downloads/mtcnn-master/anno.txt"
# with open(anno_file, 'r') as f:
#     annotations = f.readlines()
#
# with open('/Users/qiuxiaocong/Downloads/mtcnn2/imglists/anno.txt','w') as f:
#     for i in range(100):
#         index_ = np.random.choice(10000)
#         f.write(annotations[index_])

# with open('/Users/qiuxiaocong/Downloads/mtcnn2/imglists/anno.txt','r') as f:
#     annotations = f.readlines()
# print('num is :{}'.format(len(annotations)))
# annotation0 = annotations[10]
# annotation0 = annotation0.strip().split(' ')
# print(annotation0)
# print('-----------------------------')
# im_path = annotation0[0]  # 图片路径
#
# print(im_path)
# print('-----------------------------')
#
# bbox = list(map(float, annotation0[1:]))
# print(bbox)
# print('-----------------------------')
#
# boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
# print(boxes)
# print('-----------------------------')
# # # print('num is :{}'.format(len(annotations)))

#
# def load_gt_mat_to_lists(gt_dir):
#     gt_mat = loadmat(op.join(gt_dir, 'wider_face_val.mat'))
#     easy_mat = loadmat(op.join(gt_dir, 'wider_easy_val.mat'))
#     medium_mat = loadmat(op.join(gt_dir, 'wider_medium_val.mat'))
#     hard_mat = loadmat(op.join(gt_dir, 'wider_hard_val.mat'))
#     event_list = [_[0][0] for _ in gt_mat['event_list']]
#     file_list = []
#     facebox_list = []
#     easy_list = []
#     medium_list = []
#     hard_list = []
#     for file_list_per_event, box_list_per_event, easy_list_per_event, median_list_per_event, hard_list_per_event in zip(gt_mat['file_list'], gt_mat['face_bbx_list'], easy_mat['gt_list'], medium_mat['gt_list'], hard_mat['gt_list']):
#         file_list.append([_[0][0] for _ in file_list_per_event[0]])
#         facebox_list.append([_[0] for _ in box_list_per_event[0]])
#         easy_list.append([_[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist() for _ in easy_list_per_event[0]])
#         medium_list.append([_[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist() for _ in median_list_per_event[0]])
#         hard_list.append([_[0].tolist() if not _[0].tolist() else np.concatenate(_[0]).tolist() for _ in hard_list_per_event[0]])
#         print('file_list is:{}'.format(file_list))
#         print('facebox list is:{}'.format(facebox_list))
#         print('easy_list is:{}'.format(easy_list))
#         print('medium_list is:{}'.format(medium_list))
#         print('hard_list is:{}'.format(hard_list))
#         time.sleep(1000000)
#     set_gt_lists = [easy_list, medium_list, hard_list]
#     return event_list, file_list, facebox_list, set_gt_lists
#
#
# a = load_gt_mat_to_lists('/Users/qiuxiaocong/Downloads/eval_tools/ground_truth')

##########  处理lst文件 ################
# with open('/Users/qiuxiaocong/Downloads/img_cut_celeba_all.txt','r') as f:
#     annos = f.readlines()
#
# print(annos[0:5])
#
# idx_ = []
# landmark = []
# hash_map = {}
# for anno in annos:
#     anno = list(anno.strip().split(' '))
#     idx = anno[0]
#     # idx_.append(anno[0])
#     land = '\t'.join(anno[1:])
#     # landmark.append(land)
#     hash_map[idx] = land
#
# # print(hash_map)
# # print(landmark)
#
# with open('/Users/qiuxiaocong/Downloads/my_test.lst','r') as f:
#     lsts = f.readlines()
# # print(lsts[0:5])
# new_lsts = []
# for lst in lsts:
#     lst = list(lst.strip().split('\t'))
#     lst[1] = hash_map[lst[-1]]
#     lst.insert(1, str(2))
#     lst.insert(2, str(14))
#     new_lsts.append(lst)
#
# with open('/Users/qiuxiaocong/Downloads/my_test1.lst','w') as f:
#     for i in new_lsts:
#         to_write = '\t'.join(i)
#         f.write(to_write)
#         f.write('\n')
##########################################



# with open('/Users/qiuxiaocong/Downloads/celeba1.lst', 'r') as f:
#     annos = f.readlines()
#
# print(annos[0:5])
# import mxnet

import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import cv2
import time


def transform(im):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :return: [batch, channel, height, width]
    """
    im_tensor = im.transpose(2, 0, 1)
    im_tensor = im_tensor[np.newaxis, :]
    im_tensor = (im_tensor - 127.5)*0.0078125
    return im_tensor


# img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000001.jpg')
#
# print(img.shape)
# print(img[0])
# print(len(img))
# print(len(img[0]))
# cv2.imshow(img)


# train_iter = mx.image.ImageIter(batch_size=5, path_imgrec='/Users/qiuxiaocong/Downloads/my_test1.rec',
#                                 path_imgidx='/Users/qiuxiaocong/Downloads/my_test1.idx', shuffle=False,
#                                 data_shape=(3, 112, 112), label_width=16)
"""
train_iter = mx.recordio.MXRecordIO('/Users/qiuxiaocong/Downloads/my_test1.rec', 'r')
# print(train_iter)

# for i in range(1):
item = train_iter.read()
header, s = mx.recordio.unpack_img(item)

print(header)
print(s)
# cv2.imshow('a',s)
# cv2.waitKey(0)
label_ = list(list(header)[1])[2:]
# label_ = header.label()
print(label_)

print('------------------------')
train_iter.close()

img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000005.jpg')
print(img)
# cv2.imshow('a',img)
# cv2.waitKey(0)
"""
"""
time1 = time.time()
train_iter = mx.recordio.MXIndexedRecordIO('/Users/qiuxiaocong/Downloads/my_test1.idx',
                                           '/Users/qiuxiaocong/Downloads/my_test1.rec', 'r')
time2 = time.time()

items = []
for i in range(10):
    item = train_iter.read_idx(i)
    print(type(item))
    items.append(item)
print("chang du is :{}".format(len(items)))
    # header, s = mx.recordio.unpack_img(item)


time3 = time.time()
print(type(items[0]))
# a = items[0]
header, s = mx.recordio.unpack_img(items[0])
print(header)
print(s)
time4 = time.time()

print(time2-time1)
print(time3-time2)
# print(time4-time3)

# print(header)
# print(s)
# cv2.imshow('a',s)
# cv2.waitKey(0)
# label_ = list(list(header)[1])[2:]
# # label_ = header.label()
# print(label_)
#
# print('------------------------')
# train_iter.close()

time5 = time.time()
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000001.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000002.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000003.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000004.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000005.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000006.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000007.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000008.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000009.jpg')
img = cv2.imread('/Users/qiuxiaocong/Downloads/test/000010.jpg')
time6 = time.time()
print(time6-time5)
# print(img==s)
"""

# from multiprocessing import Pool
#
# def f(x):
#     return x*x
#
# if __name__ == '__main__':
#     p = Pool(2)
#     print(p.map(f,[1,2,3]))

# import multiprocessing as mp
#
# def foo(q):
#     q.put('hello')
#
# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     q = mp.Queue()
#     p = mp.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()





def test():
    path_discroc = os.path.join('/DATA/disk1/qxc/val_data', 'pnet', '',
                                '{}DiscROC.txt'.format('pnet_given'))
    print(path_discroc)


if __name__ == '__main__':

    # images = get_image_paths()      # 已经有缺省参数
    #
    # pool = Pool(24)
    # pool.map(resize_image, images)  # 注意map用法，是multiprocessing.dummy.Pool的方法
    # pool.close()
    # pool.join()

    test()












