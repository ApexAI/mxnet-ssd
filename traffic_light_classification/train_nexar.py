import os
import argparse
import logging
import sys
sys.path.append("../symbol")
import resnet
import mobilenet

import mxnet as mx

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--resume', dest='resume', help='resume from epoch',
                    default=-1, type=int)
parser.add_argument('--prefix', dest='prefix', help='model prefix',
                    default="resnet_nexar", type=str)

parser.add_argument('--train_rec', dest='train_rec', help='train file to use',
                    default="../data/nexar/nexar_train_noisy.rec", type=str)

parser.add_argument('--val_rec', dest='val_rec', help='val file to use',
                    default="../data/nexar/nexar_val_noisy.rec", type=str)


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
    pargs = parser.parse_args()
    sym = resnet.get_symbol(4,18,"3,32,32") if(pargs.net=="resnet18") else mobilenet.get_symbol(4,alpha=0.50)


    logging.getLogger().setLevel(logging.DEBUG)

    train_iter = mx.io.ImageRecordIter(
      path_imgrec=pargs.train_rec, data_name="data", label_name="softmax_label",
      batch_size=128, data_shape=(3,32,32), shuffle=True, rand_crop=True, rand_mirror=True)

    valid_iter = mx.io.ImageRecordIter(
      path_imgrec=pargs.val_rec, data_name="data", label_name="softmax_label",
    batch_size=128, data_shape=(3,32,32))

    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=100)
    epoch_end_callback = mx.callback.do_checkpoint(pargs.prefix)

    mod = mx.mod.Module(symbol=sym, context=(mx.gpu(0)))
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    if(pargs.resume>-1):
        print("resuming from %d"%pargs.resume)
        _, args, auxs = mx.model.load_checkpoint(pargs.prefix, pargs.resume)
        epoch=pargs.resume
    else:
        print("training from scratch")
        args = None
        auxs = None
        epoch=0

    learning_rate, lr_scheduler = get_lr_scheduler(0.005, '80, 160',
                                                   0.1, 48638, 128, epoch)

    optimizer_params={'learning_rate':learning_rate,
                      'momentum':0.9,
                      'wd':0.0005,
                      'lr_scheduler':lr_scheduler,
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