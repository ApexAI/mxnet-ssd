# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def get_symbol_original(num_classes, **kwargs):
    data = mx.symbol.Variable(name="data") # 224
    conv_1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1") # 224/112
    conv_2_dw = Conv(conv_1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112/112
    conv_2 = Conv(conv_2_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112/112
    conv_3_dw = Conv(conv_2, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 112/56
    conv_3 = Conv(conv_3_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56/56
    conv_4_dw = Conv(conv_3, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56/56
    conv_4 = Conv(conv_4_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56/56
    conv_5_dw = Conv(conv_4, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 56/28
    conv_5 = Conv(conv_5_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28/28
    conv_6_dw = Conv(conv_5, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28/28
    conv_6 = Conv(conv_6_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28/28
    conv_7_dw = Conv(conv_6, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 28/14
    conv_7 = Conv(conv_7_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14/14

    conv_8_dw = Conv(conv_7, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14/14
    conv_8 = Conv(conv_8_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14/14
    conv_9_dw = Conv(conv_8, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14/14
    conv_9 = Conv(conv_9_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14/14
    conv_10_dw = Conv(conv_9, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14/14
    conv_10 = Conv(conv_10_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14/14
    conv_11_dw = Conv(conv_10, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14/14
    conv_11 = Conv(conv_11_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14/14
    conv_12_dw = Conv(conv_11, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14/14
    conv_12 = Conv(conv_12_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14/14

    conv_13_dw = Conv(conv_12, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 14/7
    conv_13 = Conv(conv_13_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7/7
    conv_14_dw = Conv(conv_13, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7/7
    conv_14 = Conv(conv_14_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7

    pool = mx.sym.Pooling(data=conv_14, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool", global_pool=True)
    flatten = mx.sym.Flatten(data=pool, name="flatten")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax

def get_symbol(num_classes, **kwargs):
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv1_scale = conv1_bn
    relu1 = mx.symbol.Activation(name='relu1', data=conv1_scale , act_type='relu')

    conv2_1_dw = mx.symbol.Convolution(name='conv2_1_dw', data=relu1 , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=32)
    conv2_1_dw_bn = mx.symbol.BatchNorm(name='conv2_1_dw_bn', data=conv2_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_1_dw_scale = conv2_1_dw_bn
    relu2_1_dw = mx.symbol.Activation(name='relu2_1_dw', data=conv2_1_dw_scale , act_type='relu')

    conv2_1_sep = mx.symbol.Convolution(name='conv2_1_sep', data=relu2_1_dw , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_1_sep_bn = mx.symbol.BatchNorm(name='conv2_1_sep_bn', data=conv2_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_1_sep_scale = conv2_1_sep_bn
    relu2_1_sep = mx.symbol.Activation(name='relu2_1_sep', data=conv2_1_sep_scale , act_type='relu')

    conv2_2_dw = mx.symbol.Convolution(name='conv2_2_dw', data=relu2_1_sep , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=64)
    conv2_2_dw_bn = mx.symbol.BatchNorm(name='conv2_2_dw_bn', data=conv2_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_2_dw_scale = conv2_2_dw_bn
    relu2_2_dw = mx.symbol.Activation(name='relu2_2_dw', data=conv2_2_dw_scale , act_type='relu')

    conv2_2_sep = mx.symbol.Convolution(name='conv2_2_sep', data=relu2_2_dw , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv2_2_sep_bn = mx.symbol.BatchNorm(name='conv2_2_sep_bn', data=conv2_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_2_sep_scale = conv2_2_sep_bn
    relu2_2_sep = mx.symbol.Activation(name='relu2_2_sep', data=conv2_2_sep_scale , act_type='relu')

    conv3_1_dw = mx.symbol.Convolution(name='conv3_1_dw', data=relu2_2_sep , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=128)
    conv3_1_dw_bn = mx.symbol.BatchNorm(name='conv3_1_dw_bn', data=conv3_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_1_dw_scale = conv3_1_dw_bn
    relu3_1_dw = mx.symbol.Activation(name='relu3_1_dw', data=conv3_1_dw_scale , act_type='relu')

    conv3_1_sep = mx.symbol.Convolution(name='conv3_1_sep', data=relu3_1_dw , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_1_sep_bn = mx.symbol.BatchNorm(name='conv3_1_sep_bn', data=conv3_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_1_sep_scale = conv3_1_sep_bn
    relu3_1_sep = mx.symbol.Activation(name='relu3_1_sep', data=conv3_1_sep_scale , act_type='relu')

    conv3_2_dw = mx.symbol.Convolution(name='conv3_2_dw', data=relu3_1_sep , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=128)
    conv3_2_dw_bn = mx.symbol.BatchNorm(name='conv3_2_dw_bn', data=conv3_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_2_dw_scale = conv3_2_dw_bn
    relu3_2_dw = mx.symbol.Activation(name='relu3_2_dw', data=conv3_2_dw_scale , act_type='relu')

    conv3_2_sep = mx.symbol.Convolution(name='conv3_2_sep', data=relu3_2_dw , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv3_2_sep_bn = mx.symbol.BatchNorm(name='conv3_2_sep_bn', data=conv3_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_2_sep_scale = conv3_2_sep_bn
    relu3_2_sep = mx.symbol.Activation(name='relu3_2_sep', data=conv3_2_sep_scale , act_type='relu')

    conv4_1_dw = mx.symbol.Convolution(name='conv4_1_dw', data=relu3_2_sep , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=256)
    conv4_1_dw_bn = mx.symbol.BatchNorm(name='conv4_1_dw_bn', data=conv4_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_1_dw_scale = conv4_1_dw_bn
    relu4_1_dw = mx.symbol.Activation(name='relu4_1_dw', data=conv4_1_dw_scale , act_type='relu')

    conv4_1_sep = mx.symbol.Convolution(name='conv4_1_sep', data=relu4_1_dw , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_1_sep_bn = mx.symbol.BatchNorm(name='conv4_1_sep_bn', data=conv4_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_1_sep_scale = conv4_1_sep_bn
    relu4_1_sep = mx.symbol.Activation(name='relu4_1_sep', data=conv4_1_sep_scale , act_type='relu')

    # 28x28 -> 14x14
    conv4_2_dw = mx.symbol.Convolution(name='conv4_2_dw', data=relu4_1_sep , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=256)
    conv4_2_dw_bn = mx.symbol.BatchNorm(name='conv4_2_dw_bn', data=conv4_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_2_dw_scale = conv4_2_dw_bn
    relu4_2_dw = mx.symbol.Activation(name='relu4_2_dw', data=conv4_2_dw_scale , act_type='relu')

    conv4_2_sep = mx.symbol.Convolution(name='conv4_2_sep', data=relu4_2_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv4_2_sep_bn = mx.symbol.BatchNorm(name='conv4_2_sep_bn', data=conv4_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv4_2_sep_scale = conv4_2_sep_bn
    relu4_2_sep = mx.symbol.Activation(name='relu4_2_sep', data=conv4_2_sep_scale , act_type='relu')

    conv5_1_dw = mx.symbol.Convolution(name='conv5_1_dw', data=relu4_2_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
    conv5_1_dw_bn = mx.symbol.BatchNorm(name='conv5_1_dw_bn', data=conv5_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_1_dw_scale = conv5_1_dw_bn
    relu5_1_dw = mx.symbol.Activation(name='relu5_1_dw', data=conv5_1_dw_scale , act_type='relu')

    conv5_1_sep = mx.symbol.Convolution(name='conv5_1_sep', data=relu5_1_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_1_sep_bn = mx.symbol.BatchNorm(name='conv5_1_sep_bn', data=conv5_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_1_sep_scale = conv5_1_sep_bn
    relu5_1_sep = mx.symbol.Activation(name='relu5_1_sep', data=conv5_1_sep_scale , act_type='relu')

    conv5_2_dw = mx.symbol.Convolution(name='conv5_2_dw', data=relu5_1_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
    conv5_2_dw_bn = mx.symbol.BatchNorm(name='conv5_2_dw_bn', data=conv5_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_2_dw_scale = conv5_2_dw_bn
    relu5_2_dw = mx.symbol.Activation(name='relu5_2_dw', data=conv5_2_dw_scale , act_type='relu')

    conv5_2_sep = mx.symbol.Convolution(name='conv5_2_sep', data=relu5_2_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_2_sep_bn = mx.symbol.BatchNorm(name='conv5_2_sep_bn', data=conv5_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_2_sep_scale = conv5_2_sep_bn
    relu5_2_sep = mx.symbol.Activation(name='relu5_2_sep', data=conv5_2_sep_scale , act_type='relu')

    conv5_3_dw = mx.symbol.Convolution(name='conv5_3_dw', data=relu5_2_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
    conv5_3_dw_bn = mx.symbol.BatchNorm(name='conv5_3_dw_bn', data=conv5_3_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_3_dw_scale = conv5_3_dw_bn
    relu5_3_dw = mx.symbol.Activation(name='relu5_3_dw', data=conv5_3_dw_scale , act_type='relu')

    conv5_3_sep = mx.symbol.Convolution(name='conv5_3_sep', data=relu5_3_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_3_sep_bn = mx.symbol.BatchNorm(name='conv5_3_sep_bn', data=conv5_3_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_3_sep_scale = conv5_3_sep_bn
    relu5_3_sep = mx.symbol.Activation(name='relu5_3_sep', data=conv5_3_sep_scale , act_type='relu')

    conv5_4_dw = mx.symbol.Convolution(name='conv5_4_dw', data=relu5_3_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
    conv5_4_dw_bn = mx.symbol.BatchNorm(name='conv5_4_dw_bn', data=conv5_4_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_4_dw_scale = conv5_4_dw_bn
    relu5_4_dw = mx.symbol.Activation(name='relu5_4_dw', data=conv5_4_dw_scale , act_type='relu')

    conv5_4_sep = mx.symbol.Convolution(name='conv5_4_sep', data=relu5_4_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_4_sep_bn = mx.symbol.BatchNorm(name='conv5_4_sep_bn', data=conv5_4_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_4_sep_scale = conv5_4_sep_bn
    relu5_4_sep = mx.symbol.Activation(name='relu5_4_sep', data=conv5_4_sep_scale , act_type='relu')

    conv5_5_dw = mx.symbol.Convolution(name='conv5_5_dw', data=relu5_4_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
    conv5_5_dw_bn = mx.symbol.BatchNorm(name='conv5_5_dw_bn', data=conv5_5_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_5_dw_scale = conv5_5_dw_bn
    relu5_5_dw = mx.symbol.Activation(name='relu5_5_dw', data=conv5_5_dw_scale , act_type='relu')

    conv5_5_sep = mx.symbol.Convolution(name='conv5_5_sep', data=relu5_5_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_5_sep_bn = mx.symbol.BatchNorm(name='conv5_5_sep_bn', data=conv5_5_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_5_sep_scale = conv5_5_sep_bn
    relu5_5_sep = mx.symbol.Activation(name='relu5_5_sep', data=conv5_5_sep_scale , act_type='relu')

    conv5_6_dw = mx.symbol.Convolution(name='conv5_6_dw', data=relu5_5_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=512)
    conv5_6_dw_bn = mx.symbol.BatchNorm(name='conv5_6_dw_bn', data=conv5_6_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_6_dw_scale = conv5_6_dw_bn
    relu5_6_dw = mx.symbol.Activation(name='relu5_6_dw', data=conv5_6_dw_scale , act_type='relu')

    conv5_6_sep = mx.symbol.Convolution(name='conv5_6_sep', data=relu5_6_dw , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv5_6_sep_bn = mx.symbol.BatchNorm(name='conv5_6_sep_bn', data=conv5_6_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv5_6_sep_scale = conv5_6_sep_bn
    relu5_6_sep = mx.symbol.Activation(name='relu5_6_sep', data=conv5_6_sep_scale , act_type='relu')

    conv6_dw = mx.symbol.Convolution(name='conv6_dw', data=relu5_6_sep , num_filter=1024, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=1024)
    conv6_dw_bn = mx.symbol.BatchNorm(name='conv6_dw_bn', data=conv6_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_dw_scale = conv6_dw_bn
    relu6_dw = mx.symbol.Activation(name='relu6_dw', data=conv6_dw_scale , act_type='relu')

    conv6_sep = mx.symbol.Convolution(name='conv6_sep', data=relu6_dw , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv6_sep_bn = mx.symbol.BatchNorm(name='conv6_sep_bn', data=conv6_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv6_sep_scale = conv6_sep_bn
    relu6_sep = mx.symbol.Activation(name='relu6_sep', data=conv6_sep_scale , act_type='relu')

    pool6 = mx.symbol.Pooling(name='pool6', data=relu6_sep , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    fc7 = mx.symbol.Convolution(name='fc7', data=pool6 , num_filter=num_classes, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.symbol.Flatten(data=fc7, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    return softmax
