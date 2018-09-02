import mxnet as mx
import  vgg16_reduced_clustered

def get_symbol(*args, **kwargs):
    return vgg16_reduced_clustered.get_symbol(shrink=4, **kwargs)