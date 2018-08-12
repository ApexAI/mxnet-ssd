import mxnet as mx



def get_symbol(*args, **kwargs):
    return mx.sym.load("symbol/vgg16_finetuned-symbol.json")