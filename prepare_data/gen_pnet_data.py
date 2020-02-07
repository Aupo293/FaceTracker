import numpy as np
import cv2
import os
import numpy.random as npr
from prepare_data.utils import IoU

# anno_file = "./wider_annotations/anno.txt"
# im_dir = "/home/seanlx/Dataset/wider_face/WIDER_train/images"
# neg_save_dir = "/data3/seanlx/mtcnn1/12/negative"
# pos_save_dir = "/data3/seanlx/mtcnn1/12/positive"
# part_save_dir = "/data3/seanlx/mtcnn1/12/part"

# 整个train训练集的annotation
anno_file = "/Users/qiuxiaocong/Downloads/mtcnn-master/anno.txt"
im_dir = "/Users/qiuxiaocong/Downloads/WIDER_train/images"
neg_save_dir = "/Users/qiuxiaocong/Downloads/mtcnn1/12/negative"
pos_save_dir = "/Users/qiuxiaocong/Downloads/mtcnn1/12/positive"
part_save_dir = "/Users/qiuxiaocong/Downloads/mtcnn1/12/part"

# save_dir = "./pnet"
save_dir = "/Users/qiuxiaocong/Downloads/mtcnn1/pnet"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')   # pos
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')   # neg
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')  # part

with open(anno_file, 'r') as f:   # 所有的train样本的annotation
    annotations = f.readlines()

num = len(annotations)  # train样本个数
print("%d pics in total" % num)

p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # don't care (part?)
idx = 0   # 图片索引
box_idx = 0  # 所有框的个数
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]  # 图片路径
    # bbox = map(float, annotation[1:])
    bbox = list(map(float, annotation[1:]))

    # one image 的所有GT
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    # 读取该image
    # print(im_dir, im_path + '.jpg')
    # img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    img = cv2.imread(os.path.join(im_path))

    # img索引加一
    # cv2.imshow('a',img)
    # cv2.waitKey()
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape  # 输入图片的高/宽/通道数

    neg_num = 0
    # 每张image生成50个negative sample[不依赖于GT产生] 即其产生的neg sample与GT可能没有IOU值
    while neg_num < 50:
        # 从[12,min(width, height) / 2)范围中生成一个随机数
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        # 仅仅是box的左上角x/y坐标和右下角x/y坐标,还不是图片
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)
        # 从原图中crop得到的图片
        cropped_im = img[ny: ny + size, nx: nx + size, :]
        # 将crop部分resize成 12×12,输入P_NET
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        # IOU最大值都小于0.3的图片归入negative部分
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1   # 用于记录总共多少negative sample
            neg_num += 1  # 用于每张图片的50个negative sample的选取

    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        # 这里考虑得到少许(5个)的neg sample,其与GT有 IOU,但值小于0.3
        for i in range(5):
            size = npr.randint(12,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            print(ny1,nx1,size)
            # cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            cropped_im = img[int(ny1): int(ny1) + size, int(nx1): int(nx1) + size, :]

            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("12/negative/%s"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples and part faces

        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            print(ny1, nx1, ny2, nx2)
            # cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]

            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("12/positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
