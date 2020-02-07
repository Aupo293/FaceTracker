from prepare_data.wider_annotations.wider_loader import WIDER
import cv2
import time

"""
因为数据集的训练标签是MATLAB格式的，所以先转换成txt
"""

# wider face original images path 训练图片存放路径
# path_to_image = '/home/seanlx/Dataset/wider_face/WIDER_train/images'
# path_to_image = '/DATA/disk1/qxc/WIDER_train/images'
path_to_image = '/Users/qiuxiaocong/Downloads/WIDER_val/images'



# matlab file path
# file_to_label = './wider_face_train.mat'
# file_to_label = '/DATA/disk1/qxc/mtcnn/prepare_data/wider_annotations/wider_face_train.mat'
file_to_label = '/Users/qiuxiaocong/Downloads/mtcnn/prepare_data/wider_annotations/wider_face_val.mat'


# target file path  anno存放的是Ground Truth box的左上角x/y坐标 和 右下角x/y坐标
target_file = '/Users/qiuxiaocong/Downloads/anno_val.txt'
# target_file = '/DATA/disk1/qxc/mtcnn1/anno.txt'


wider = WIDER(file_to_label, path_to_image)


line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%ld'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)


