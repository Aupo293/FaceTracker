import logging
import mxnet as mx
import core.metric as metric
from mxnet.module.module import Module
from core.loader import ImageLoader
from core.imdb import IMDB
from config import config
from tools.load_model import load_param


def train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, imdb, net=12, frequent=50, initialize=True, base_lr=0.01):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   # 记录到标准输出


    # 训练数据
    train_data = ImageLoader(imdb, net, config.BATCH_SIZE, shuffle=True, ctx=ctx)

    if not initialize:  # 如果非初始化 加载参数
        args, auxs = load_param(pretrained, epoch, convert=True)

    if initialize:
        print("init weights and bias:")
        data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
        arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

        # 权重初始化 Xavier初始化器
        init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        args = dict()   # 模型参数以及网络权重字典
        auxs = dict()   # 模型参数以及一些附加状态的字典

        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue

            print('init', k)

            args[k] = mx.nd.zeros(arg_shape_dict[k])
            init(k, args[k])
            if k.startswith('fc'):
                args[k][:] /= 10

            '''
            if k.endswith('weight'):
                if k.startswith('conv'):
                    args[k] = mx.random.normal(loc=0, scale=0.001, shape=arg_shape_dict[k])
                else:
                    args[k] = mx.random.normal(loc=0, scale=0.01, shape=arg_shape_dict[k])
            else: # bias
                args[k] = mx.nd.zeros(shape=arg_shape_dict[k])
            '''

        for k in sym.list_auxiliary_states():
            auxs[k] = mx.nd.zeros()
            init(k, auxs[k])

    lr_factor = 0.1
    lr_epoch = config.LR_EPOCH
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(imdb) / config.BATCH_SIZE) for epoch in lr_epoch_diff]
    print('lr:{},lr_epoch:{},lr_epoch_diff:{}'.format(lr, lr_epoch,lr_epoch_diff))
    # print('lr', lr, 'lr_epoch', lr_epoch, 'lr_epoch_diff', lr_epoch_diff)

    # MXNet设置动态学习率,经过lr_iters次更新后,学习率变为lr*lr_factor
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    # 作用是每隔多少个batch显示一次结果
    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=frequent)
    # 作用是每隔period个epoch保存训练得到的模型
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # 调用评价函数类
    eval_metrics = mx.metric.CompositeEvalMetric()
    metric1 = metric.Accuracy()
    metric2 = metric.LogLoss()
    metric3 = metric.BBOX_MSE()
    # 使用add方法添加评价函数类
    for child_metric in [metric1, metric2, metric3]:
        eval_metrics.add(child_metric)
    # 优化相关参数
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.00001,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': 5}
    # 创建一个可训练的模块
    mod = Module(sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx)
    # 训练模型
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=end_epoch)


"""
def train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, imdb, net=12, frequent=50, initialize=True, base_lr=0.01):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   # 记录到标准输出

    # 训练数据
    train_data = ImageLoader(imdb, net, config.BATCH_SIZE, shuffle=True, ctx=ctx)

    if not initialize:  # 如果非初始化 加载参数
        args, auxs = load_param(pretrained, epoch, convert=True)

    if initialize:
        print("init weights and bias:")
        data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
        arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

        # 权重初始化 Xavier初始化器
        init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        args = dict()   # 模型参数以及网络权重字典
        auxs = dict()   # 模型参数以及一些附加状态的字典

        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue

            print('init', k)

            args[k] = mx.nd.zeros(arg_shape_dict[k])
            init(k, args[k])
            if k.startswith('fc'):
                args[k][:] /= 10

            '''
            if k.endswith('weight'):
                if k.startswith('conv'):
                    args[k] = mx.random.normal(loc=0, scale=0.001, shape=arg_shape_dict[k])
                else:
                    args[k] = mx.random.normal(loc=0, scale=0.01, shape=arg_shape_dict[k])
            else: # bias
                args[k] = mx.nd.zeros(shape=arg_shape_dict[k])
            '''

        for k in sym.list_auxiliary_states():
            auxs[k] = mx.nd.zeros()
            init(k, auxs[k])

    lr_factor = 0.1
    lr_epoch = config.LR_EPOCH
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(imdb) / config.BATCH_SIZE) for epoch in lr_epoch_diff]
    print('lr:{},lr_epoch:{},lr_epoch_diff:{}'.format(lr, lr_epoch,lr_epoch_diff))
    # print('lr', lr, 'lr_epoch', lr_epoch, 'lr_epoch_diff', lr_epoch_diff)

    # MXNet设置动态学习率,经过lr_iters次更新后,学习率变为lr*lr_factor
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    # 作用是每隔多少个batch显示一次结果
    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=frequent)
    # 作用是每隔period个epoch保存训练得到的模型
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # 调用评价函数类
    eval_metrics = mx.metric.CompositeEvalMetric()
    metric1 = metric.Accuracy()
    metric2 = metric.LogLoss()
    metric3 = metric.BBOX_MSE()
    # 使用add方法添加评价函数类
    for child_metric in [metric1, metric2, metric3]:
        eval_metrics.add(child_metric)
    # 优化相关参数
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.00001,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0}
    # 创建一个可训练的模块
    mod = Module(sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx)
    # 训练模型
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=end_epoch)
"""