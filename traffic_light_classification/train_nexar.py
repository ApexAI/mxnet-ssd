import os
import argparse
import logging
import sys
sys.path.append("../symbol")
import resnet
import mobilenet
import densenet
import re
import mxnet as mx

parser = argparse.ArgumentParser(description='Train a traffic light classifier')
parser.add_argument('--resume', dest='resume', help='resume from epoch',
                    default=-1, type=int)
parser.add_argument('--prefix', dest='prefix', help='model prefix',
                    default="resnet_nexar", type=str)

parser.add_argument('--train_rec', dest='train_rec', help='train file to use',
                    default="../data/nexar/nexar_train_noisy.rec", type=str)

parser.add_argument('--val_rec', dest='val_rec', help='val file to use',
                    default="../data/nexar/nexar_val_noisy.rec", type=str)
parser.add_argument('--lr', dest='lr', help='learning rate',
                    default=0.005, type=float)

parser.add_argument('--batch-size', dest='batch_size', help='batch size',
                    default=64, type=int)

parser.add_argument('--freeze', dest='freeze', help='freeze some layers for finetuning',
                    default=False, type=bool)

parser.add_argument('--net', dest='net', help='val file to use',
                    default="resnet18", type=str)

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]

    lr = learning_rate
    epoch_size = num_example // batch_size
    for s in iter_refactor:
        if begin_epoch >= s:
            lr *= lr_refactor_ratio
    if lr != learning_rate:
        logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
    if not steps:
        return (lr, None)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
    return (lr, lr_scheduler)




if __name__ == '__main__':
    # download data

    patterns={}
    patterns['densenet30']= "^(conv0_|DBstage1_unit._conv1_|DBstage2_unit._conv1_).*"
    patterns ['resnet18']="^(conv0_|stage1_unit._conv._|stage2_unit._conv._|stage3_unit._conv._|stage4_unit1_conv._).*"

    pargs = parser.parse_args()

    if(pargs.resume>-1):
        print("resuming from %d"%pargs.resume)
        sym, args, auxs = mx.model.load_checkpoint(pargs.prefix, pargs.resume)
        epoch=pargs.resume
    else:
        if(pargs.net=="resnet18"):
            sym=resnet.get_symbol(4,18,"3,32,32")
        elif (pargs.net=="mobilenet050"):
            sym=mobilenet.get_symbol(4,alpha=0.50)
        elif (pargs.net=="densenet30"):
            sym=densenet.DenseNet(units=[9,9,9], num_stage=3, growth_rate=12, num_class=4, data_type="nexar",
                              bottle_neck=False)
        args = None
        auxs = None
        epoch=0

    if(pargs.freeze):
        re_prog = re.compile(patterns[pargs.net])
        fixed_param_names = [name for name in sym.list_arguments() if re_prog.match(name)]
        if fixed_param_names:
            print("Freezed parameters: [" + ','.join(fixed_param_names) + ']')
    else:
        fixed_param_names=None

    logging.getLogger().setLevel(logging.DEBUG)

    train_iter = mx.io.ImageRecordIter(
      path_imgrec=pargs.train_rec, data_name="data", label_name="softmax_label",
      batch_size=pargs.batch_size, data_shape=(3,32,32), shuffle=True, rand_crop=True, rand_mirror=True, max_rotate_angle=20, max_shear_ratio=0.1,
        random_h=20,random_s=30,random_l=30)

    valid_iter = mx.io.ImageRecordIter(
      path_imgrec=pargs.val_rec, data_name="data", label_name="softmax_label",
    batch_size=pargs.batch_size, data_shape=(3,32,32))

    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=100)
    epoch_end_callback = mx.callback.do_checkpoint(pargs.prefix)

    mod = mx.mod.Module(symbol=sym, context=(mx.gpu(0)), fixed_param_names=fixed_param_names)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)



    learning_rate, lr_scheduler = get_lr_scheduler(pargs.lr, '80, 160',
                                                   0.1, 48638, 128, epoch)

    optimizer_params={'learning_rate':pargs.lr,
                      'momentum':0.9,
                      'wd':0.0005,
                      #'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0 }

    mod.fit(train_iter, eval_data=valid_iter, optimizer='sgd',
            optimizer_params=optimizer_params,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            num_epoch=100,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch
            )

    print(mod.score(valid_iter, mx.metric.Accuracy()))