import mxnet as mx
import cv2
import numpy as np
# define a simple data batch
import time
from collections import namedtuple
import argparse

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--prefix', dest='prefix', help='train list to use',
                    default="trained_models/mobilenet050", type=str)
parser.add_argument('--epoch', dest='epoch', help='train list to use',
                    default=28, type=int)
parser.add_argument('--image_path', dest='image_path', help='train list to use',
                    default="../data/demo/65.png", type=str)
Batch = namedtuple('Batch', ['data'])





def image_forward(image_names,prefix,epoch, batch_size=1, ctx=mx.cpu(0)):
    # img = cv2.cvtColor(cv2.imread("/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_lights/28862_0.jpg"), cv2.COLOR_BGR2RGB)
    sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)
    labels = {0: "green", 1: "yellow", 2: "red", 3: "none"}

    num_img=len(image_names)
    batches= [image_names[i:i + batch_size] for i in range(0, num_img, batch_size)]
    mod = mx.mod.Module(symbol=sym, context=ctx)
    mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 32, 32))])
    mod.set_params(args, auxs)
    probs=[]
    durations=[]

    results=[]

    for batch in batches:
        images = []
        if(len(batch) != batch_size):
            mod = mx.mod.Module(symbol=sym, context=ctx)
            mod.bind(for_training=False, data_shapes=[('data', (len(batch), 3, 32, 32))])
            mod.set_params(args, auxs)

        for im_name in batch:

            img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)

            # img=cv2.imread("/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_lights/26925_5.jpg")
            # convert into format (batch, RGB, width, height)
            img = cv2.resize(img, (32, 32))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img[np.newaxis, :]
            images.append(img[0])
        begin = time.time()
        mod.forward(Batch([mx.nd.array(images)]))
        #probs.append(mod.get_outputs())
        probs=mod.get_outputs()[0].asnumpy()
        batch_duration=time.time() - begin

        for prob in probs:
            result = {}
            #prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            result['prediction'] = labels[a[0]]
            for i in a[0:4]:
                result[labels[i]] = prob[i]
            result["duration"] = batch_duration/len(batch)
            results.append(result)

    return  results
    # valid_iter = mx.io.ImageRecordIter(
    #    path_imgrec="../data/nexar/nexar_val.rec", data_name="data", label_name="softmax_label",
    #    batch_size=128, data_shape=(3, 32, 32))




if __name__ == '__main__':
    pargs = parser.parse_args()

    results=image_forward([pargs.image_path],pargs.prefix,pargs.epoch)

    for result in results:
        for key, value in result.iteritems():
            print("{}: {}".format(key,value))
#            print('probability=%f, class=%s' % (value, key))


