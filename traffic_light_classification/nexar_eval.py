import mxnet as mx
import argparse

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--prefix', dest='prefix', help='train list to use',
                    default="models/resnet_nexar", type=str)
parser.add_argument('--epoch', dest='epoch', help='train list to use',
                    default=10, type=int)
parser.add_argument('--rec', dest='rec', help='train list to use',
                    default="roi1.rec", type=str)

pargs = parser.parse_args()

sym, args, auxs = mx.model.load_checkpoint(pargs.prefix, pargs.epoch)

valid_iter = mx.io.ImageRecordIter(
    path_imgrec=pargs.rec, data_name="data", label_name="softmax_label",
    batch_size=128, data_shape=(3, 32, 32))


mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))

mod.bind(for_training=False,
         data_shapes=valid_iter.provide_data,
         label_shapes=valid_iter.provide_label)

mod.set_params(args, auxs)

metric = mx.metric.Accuracy()

score = mod.score(valid_iter, metric)

print(score)