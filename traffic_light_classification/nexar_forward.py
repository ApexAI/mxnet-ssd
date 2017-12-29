import mxnet as mx
import cv2
import numpy as np
# define a simple data batch

from collections import namedtuple
import argparse

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--prefix', dest='prefix', help='train list to use',
                    default="models/resnet_nexar", type=str)
parser.add_argument('--epoch', dest='epoch', help='train list to use',
                    default=0, type=int)
parser.add_argument('--image_path', dest='image_path', help='train list to use',
                    default="../data/demo/65.png", type=str)

Batch = namedtuple('Batch', ['data'])





def image_forward(fname,prefix,epoch,ctx=mx.cpu(0)):
    # img = cv2.cvtColor(cv2.imread("/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_lights/28862_0.jpg"), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    # img=cv2.imread("/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_lights/26925_5.jpg")
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (32, 32))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)

    # valid_iter = mx.io.ImageRecordIter(
    #    path_imgrec="../data/nexar/nexar_val.rec", data_name="data", label_name="softmax_label",
    #    batch_size=128, data_shape=(3, 32, 32))


    labels = {0: "green", 1: "yellow", 2: "red", 3:"none"}
    mod = mx.mod.Module(symbol=sym, context=ctx)

    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 32, 32))])

    mod.set_params(args, auxs)

    metric = mx.metric.Accuracy()

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    results={}
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    results['prediction']=labels[a[0]]
    for i in a[0:4]:
        results[labels[i]]=prob[i]
    return results


if __name__ == '__main__':
    pargs = parser.parse_args()

    result=image_forward(pargs.image_path,pargs.prefix,pargs.epoch)

    for key, value in result.iteritems():
        print("{}: {}".format(key,value))
#        print('probability=%f, class=%s' % (value, key))


