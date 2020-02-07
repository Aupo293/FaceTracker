import numpy as np
import numpy.random as npr


"""
合并 negative/positive/part 的训练样本,得到 train_xx.txt
"""

size = 24

if size == 12:
    net = "pnet"
elif size == 24:
    net = "rnet"
elif size == 48:
    net = "onet"

with open('/Users/qiuxiaocong/Downloads/mtcnn1/prepare_data/%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('/Users/qiuxiaocong/Downloads/mtcnn1/prepare_data/%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('/Users/qiuxiaocong/Downloads/mtcnn1/prepare_data/%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()


with open("/Users/qiuxiaocong/Downloads/mtcnn1/imglists/train_%s.txt"%(size), "w") as f:
    f.writelines(pos)   # pos是全部选取

    # neg sample选取约600000张, part sample选取约300000张
    # 文章比例为neg: pos: part的比例为3:1:1
    neg_keep = npr.choice(len(neg), size=6000, replace=True)
    part_keep = npr.choice(len(part), size=3000, replace=True)
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])


'-----------------------------------------------------------------'
"""
''
with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()


with open("%s/train_%s.txt"%(net, size), "w") as f:
    f.writelines(pos)
    neg_keep = npr.choice(len(neg), size=600000, replace=False)
    part_keep = npr.choice(len(part), size=300000, replace=False)
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
"""


