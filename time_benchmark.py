from object_detector import object_detector
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='Single-shot detection network demo')

parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                    default=46, type=int)
parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                    default="car_models/deploy_resnet18_2c_v2",
                    type=str)
parser.add_argument('--image', dest='image', help='image path',
                    default="data/demo/street.jpg",
                    type=str)
parser.add_argument('--ctx', dest='ctx', help='image path',
                    default="gpu",
                    type=str)
pargs = parser.parse_args()


sum=0.0

detector=object_detector.object_detector(prefix=pargs.prefix, epoch=pargs.epoch, conf_thresh=0.5, device=pargs.ctx)
im=cv2.imread("data/demo/street.jpg")
loop=1000

for i in range(loop):
    begin=time.time()
    detector.detect_objects(im)
    sum+=time.time()-begin

print("total time is: {}, average is {}".format(sum,sum/loop))