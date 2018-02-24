import mxnet as mx
import argparse
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from detect.detector import Detector
from symbol.symbol_factory import get_symbol
import os

from PIL import Image

from PIL import ImageDraw
from PIL import ImageFont


parser = argparse.ArgumentParser(description='order and save detections by confidence')

parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                    default=0, type=int)
parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                    default="resnet18",
                    type=str)

parser.add_argument('--dir', dest='dir', help='dir of images',
                    default=".",
                    type=str)
parser.add_argument('--batch-size', dest='batch_size', help='batchsize',
                    default=1, type=int)

data_shape=512
sum=0
pargs = parser.parse_args()
classes=['car','notcar']
thresh=0.2


ims=os.listdir(pargs.dir)

loadsym, args, auxs = mx.model.load_checkpoint(pargs.prefix, pargs.epoch)

net = get_symbol('resnet18', data_shape, num_classes=2, nms_thresh=0.5)

mod = mx.mod.Module(net, label_names=None, context=mx.cpu())

mod.bind(data_shapes=[('data', (pargs.batch_size, 3, data_shape, data_shape))])



font                   = cv2.FONT_HERSHEY_SIMPLEX


idx=0
mod.set_params(args, auxs)

boxes=[]

for im in ims:
    test_db = TestDB([im], root_dir=None, extension=None)
    test_iter = DetIter(test_db, 1, data_shape, is_train=False)

    num_images = test_iter._size
    result = []
    detections = []


    for pred, _, _ in mod.iter_predict(test_iter):
        detections.append(pred[0].asnumpy())

    for output in detections:
        for i in range(output.shape[0]):
            det = output[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)

    for k, det in enumerate(result):
        img = cv2.imread(im)
        orig=img
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
        # self.visualize_detection(img, det, classes, thresh)
        plt.gca().patches = []
        plt.gca().texts = []

        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(det.shape[0]):
            cls_id = int(det[i, 0])
            if cls_id >= 0:
                score = det[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(det[i, 2] * width)
                    ymin = int(det[i, 3] * height)
                    xmax = int(det[i, 4] * width)
                    ymax = int(det[i, 5] * height)

                    boxes.append((im,score,(xmin,ymin,xmax,ymax)))



def compare_score(s1, s2):
    return cmp(s1[1],s2[1])

sorted_boxes=sorted(boxes,cmp=compare_score)


for bidx, box in enumerate(sorted_boxes):
    imfile=box[0]
    score=box[1]
    (xmin,ymin,xmax,ymax)=box[2]

    img = cv2.imread(imfile)[ymin-20:ymax, xmin:xmax]
    cv2.putText(img, str(score), (5, 30), font, 0.1*height/300, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite('car_models/cropped/{}.png'.format(bidx), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


scores=np.array([s[1] for s in sorted_boxes])
mean=np.mean(scores)
variance=np.var(scores)
median=np.median(scores)


print("mean: {}, variance: {}, median, {} \n".format(mean,variance,median))

print("num elements larger than mean: {}".format(len(np.where(scores>mean)[0])))
print("mean next image: {}.png\n".format(np.where(scores>mean)[0][0]))

print("num elements larger than 0.3: {}".format(len(np.where(scores>0.3)[0])))
print("0.3 next image: {}.png\n".format(np.where(scores>0.3)[0][0]))


print("num elements larger than 0.45: {}".format(len(np.where(scores>0.45)[0])))
print("0.45 next image: {}.png\n".format(np.where(scores>0.45)[0][0]))

print("num elements larger than 0.5: {}".format(len(np.where(scores>0.5)[0])))
print("0.5 next image: {}.png\n".format(np.where(scores>0.5)[0][0]))

print("num elements larger than 0.6: {}".format(len(np.where(scores>0.6)[0])))
print("0.6 next image: {}.png\n".format(np.where(scores>0.6)[0][0]))

print("num elements larger than 0.7: {}".format(len(np.where(scores>0.7)[0])))
print("0.7 next image: {}.png\n".format(np.where(scores>0.7)[0][0]))

print("num elements larger than 0.75: {}".format(len(np.where(scores>0.75)[0])))
print("0.75 next image: {}.png\n".format(np.where(scores>0.75)[0][0]))

print("num elements larger than 0.8: {}".format(len(np.where(scores>0.8)[0])))
print("0.8 next image: {}.png\n".format(np.where(scores>0.8)[0][0]))

print("num elements larger than 0.85: {}".format(len(np.where(scores>0.85)[0])))
print("0.85 next image: {}.png\n".format(np.where(scores>0.85)[0][0]))

print("num elements larger than 0.9: {}".format(len(np.where(scores>0.9)[0])))
print("0.9 next image: {}.png\n".format(np.where(scores>0.9)[0][0]))

print("num elements larger than 0.95: {}".format(len(np.where(scores>0.95)[0])))
print("0.95 next image: {}.png\n".format(np.where(scores>0.95)[0][0]))