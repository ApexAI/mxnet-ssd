#Script to run the classifier on a single image
import mxnet as mx
import cv2
import numpy as np
import time
from collections import namedtuple
import os
import argparse


parser = argparse.ArgumentParser(description='Run prediction on a traffic light image')
parser.add_argument('--network', dest='network', help='network name: resnet18, mobilenet050 or densenet30',
                    default="resnet18", type=str)
parser.add_argument('--modelpath', dest='modelpath', help='directory containing trained models',
                    default="data", type=str)
parser.add_argument('--image_path', dest='image_path', help='image to be classified',
                    default="../data/demo/65.png", type=str)
parser.add_argument('--device', dest='dev', help='cpu or gpu to use',
                    default="gpu", type=str)


Batch = namedtuple('Batch', ['data'])
labels = {0: "green", 1: "yellow", 2: "red", 3: "none"}


#epoch numbers are hardcoded to fit 3 pretrained models
def get_model(network, rootpath):
    prefix = os.path.join(rootpath, network)
    if network=="resnet18":
        epoch=20
    elif network=="densenet30":
        epoch=9
    elif network=="mobilenet050":
        epoch=28
    else:
        raise ValueError("the network {} is not defined".format(network))
    return prefix, epoch


def image_forward(fname,network,modelpath, device):

    prefix,epoch= get_model(network,modelpath)

    ctx= mx.cpu(0) if device=="cpu" else mx.gpu(0)

    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (32, 32))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    #load the model
    sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)

    mod = mx.mod.Module(symbol=sym, context=ctx)

    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 32, 32))])

    mod.set_params(args, auxs)


    #run forward pass
    begin=time.time()
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    duration=time.time()-begin

    # get the probabilities and the prediction
    results={}
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    results['prediction']=labels[a[0]]
    for i in a[0:4]:
        results[labels[i]]=prob[i]
    results["duration"]=duration
    return results



if __name__ == '__main__':
    pargs = parser.parse_args()

    result=image_forward(pargs.image_path,pargs.network, pargs.modelpath, pargs.dev)

    print("\nprediction: {} \n".format(result['prediction']))

    print("Classification time: {}s \n".format(result['duration']))

    print("Probabilities:")
    for key, val in labels.iteritems():
        print("{}: {}".format(val,result[val]))
