import os
from evaluate.evaluate_net import evaluate_net
import logging

logging.getLogger().setLevel(logging.ERROR)
import mxnet as mx
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ctx', default="gpu", type=str,help='train folder')
args = parser.parse_args()

ctx = mx.gpu() if args.ctx =="gpu" else mx.cpu()

command_bodies_pretrained= [
    ["mobilenet", "mobilenet_apex_v2_2", 4],
    ["resnet18", "resnet18_apex_v3", 70],
    ["densenet121", "densenet_test", 56],
    ["vgg16_reduced", "vgg16_reduced_300", 27],
    ["vgg16_reduced_clustered2x", "vgg16_clustered2x", 80],
    ["vgg16_reduced_clustered4x", "vgg16_clustered4x", 84],
    ["resnet18_clustered4x", "resnet18_clustered4x", 74]
]
file_str = ""

command_bodies_scratch= [
    ["mobilenet", "mobilenet_scratch", 70],
    ["resnet18", "resnet18_apex_v3_scratch", 54],
    ["pvanet","pvanet_scratch", 87],
    ["shufflenet", "shufflenet_scratch", 66]
]


with open('dataset/names/apex_coco.names', 'r') as f:
    class_names = [l.strip() for l in f.readlines()]



def runtest(batch_size,ctx, pretrained):
    instr=""
    networks = command_bodies_pretrained if pretrained else command_bodies_scratch
    path = "trained_models/{}/".format("pretrained" if pretrained else "scratch")
    for ci, cmd in enumerate(networks):
        print"cmd is:{}".format(cmd)
        result = evaluate_net(cmd[0], "data/coco_apex_val.rec", 5,
                              None, 300, path + cmd[1], cmd[2], ctx, batch_size=batch_size,
                              nms_thresh=0.45, force_nms=False,
                              ovp_thresh=0.5, use_difficult=False, class_names=class_names,
                              voc07_metric=True, cars_only=False
                              )
        res = "\n############################################\n"\
            "Batch size: {}, device: {}, pretrained: {}\n" \
            "for network {}, acc: {}, fps {}\n" \
            "############################################\n".format(batch_size, ctx, pretrained,
                                                       cmd[1], result[1][5][1], float(1)/(result[0]/952))
        print res
        instr +=res
        nm = mx.name.NameManager().current  # hack to workaround mxnet's auto name generation
        nm._counter = {}
    return instr


batch_sizes = [1,64, 128]
training = [True,False]

for bs in batch_sizes:
    for pretrained in training:
        print (bs, ctx, pretrained)
        file_str += runtest(bs, ctx, pretrained)




with open("benchmarks_{}".format(args.ctx), 'w') as f:
    f.write(file_str)