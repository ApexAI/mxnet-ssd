import argparse
import nexar_forward
import time
import mxnet as mx
parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--prefix', dest='prefix', help='model prefix',
                    default="models/resnet_nexar_noisy", type=str)
parser.add_argument('--epoch', dest='epoch', help='epoch num',
                    default=20, type=int)
parser.add_argument('--image_path', dest='image_path', help='image path',
                    default="../data/demo/65.png", type=str)
parser.add_argument('--loop', dest='loop', help='loop length to take average',
                    default=1000, type=int)
parser.add_argument('--device', dest='device', help='cpu or gpu to use',
                    default="cpu", type=str)
pargs = parser.parse_args()
total=0
loop=pargs.loop

for i in range(loop):
    begin=time.time()
    dev=mx.cpu(0) if pargs.device=="cpu" else mx.gpu(0)
    prediction = nexar_forward.image_forward(pargs.image_path, pargs.prefix, pargs.epoch, dev)
    duration=time.time()-begin
    if(i is not 0):
        total+=duration
    print(duration)

print total/(loop-1)