import mxnet as mx
from core import negativemining
from config import config

def P_Net(mode='train'):
    """
    Proposal Network
    input shape 3 x 12 x 12
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=10, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=16, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")

    conv3 = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")

    conv4_1 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=2, name="conv4_1")
    # conv4_1对应face classification
    conv4_2 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=4, name="conv4_2")
    # conv4_2对应bounding box regression

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])

    else:
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1, label=label,
                                           multi_output=True, use_ignore=True,
                                             name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1,   name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred,
                               label=label, bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group


def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=28, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")

    fc1 = mx.symbol.FullyConnected(data=prelu3, num_hidden=128, name="fc1")
    prelu4 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu4")

    fc2 = mx.symbol.FullyConnected(data=prelu4, num_hidden=2, name="fc2")
    # fc2对应 face classification
    fc3 = mx.symbol.FullyConnected(data=prelu4, num_hidden=4, name="fc3")
    # fc3对应 bounding box regression


    cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True,
                                         name="cls_prob")
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = fc3
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=fc3, label=bbox_target,
                                                       grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group


def O_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")
    pool3 = mx.symbol.Pooling(data=prelu3, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4 = mx.symbol.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name="conv4")
    prelu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name="prelu4")

    fc1 = mx.symbol.FullyConnected(data=prelu4, num_hidden=256, name="fc1")
    prelu5 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu5")

    fc2 = mx.symbol.FullyConnected(data=prelu5, num_hidden=2, name="fc2")
    # fc2对应 face classification
    fc3 = mx.symbol.FullyConnected(data=prelu5, num_hidden=4, name="fc3")
    # fc3对应 bounding box regression

    cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True,   name="cls_prob")
    if mode == "test":
        bbox_pred = fc3
        # cls_prob = fc2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        # print('bbox_pred is:{}'.format(bbox_pred))
        # print('cls_prob is:{}'.format(cls_prob))
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=fc3, label=bbox_target,
                                                     grad_scale=1,   name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group


lnet_basenum = 32
def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1")  # 48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum,
                                     name="conv2_dw")  # 46/45
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")

    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum,
                                     num_group=lnet_basenum, name="conv3_dw")  # 45/22
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum * 2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")

    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet_basenum * 2,
                                     num_group=lnet_basenum * 2, name="conv4_dw")  # 22/21
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum * 2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum * 2,
                                     num_group=lnet_basenum * 2, name="conv5_dw")  # 21/10
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum * 4, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")

    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet_basenum * 4,
                                     num_group=lnet_basenum * 4, name="conv6_dw")  # 10/9
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum * 4, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")

    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum * 4,
                                     num_group=lnet_basenum * 4, name="conv7_dw")  # 9/4
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet_basenum * 8, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet_basenum * 8,
                                     num_group=lnet_basenum * 8, name="conv8_dw")  # 4/3
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet_basenum * 8, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet_basenum * 8,
                                     num_group=lnet_basenum * 8, name="conv9_dw")  # 3/1
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet_basenum * 8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=10, name="conv6_3")
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False, momentum=0.9)

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data=landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data=bn6_3, axis=1, begin=9, end=10)
            # pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")
            # pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")
            # pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")
            # pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")
            # pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")
            # pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")
            # pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")
            # pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")
            # pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")
            # pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")
            out = mx.symbol.Custom(pred_x1=bn_x1, pred_x2=bn_x2, pred_x3=bn_x3, pred_x4=bn_x4, pred_x5=bn_x5,
                                   pred_y1=bn_y1, pred_y2=bn_y2, pred_y3=bn_y3, pred_y4=bn_y4, pred_y5=bn_y5,
                                   target_x1=target_x1, target_x2=target_x2, target_x3=target_x3, target_x4=target_x4,
                                   target_x5=target_x5,
                                   target_y1=target_y1, target_y2=target_y2, target_y3=target_y3, target_y4=target_y4,
                                   target_y5=target_y5,
                                   op_type='negativemining_onlylandmark10', name="negative_mining")
            group = mx.symbol.Group([out])
        else:
            # landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
            #                                         grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=bn6_3, landmark_target=landmark_target,
                                   op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

#
# if __name__ == '__main__':
#     net = R_Net()
#     x = mx.sym.var('data')
#     out = net(x)
#     net.export('.')