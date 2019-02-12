import sys, os
sys.path.insert(0,'/home/yehao/caffe_SSD/python') # your caffe path

import re
import fileinput

import caffe
from caffe import layers as L

def bottleneck(net, net_bottom, prefix, input_channel, time, out, step):
    if (prefix != 'conv2'):
        net.tops[prefix+'/1x1_up'] = L.Convolution(net_bottom, num_output=input_channel * time, 
                        kernel_size=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.tops[prefix+'/1x1_up/bn'] = L.BatchNorm(net.tops[prefix+'/1x1_up'], param=[dict(lr_mult=0, decay_mult=0), 
                dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
        net.tops[prefix+'/1x1_up/scale'] = L.Scale(net.tops[prefix+'/1x1_up/bn'], param=[dict(lr_mult=1, decay_mult=0),
                dict(lr_mult=2, decay_mult=0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
        net.tops[prefix+'/1x1_up/relu6'] = L.ReLU6(net.tops[prefix+'/1x1_up/scale'], in_place=True)

        net.tops[prefix+'/3x3_dw'] = L.ConvolutionDepthwise(net.tops[prefix+'/1x1_up/relu6'], num_output=input_channel * time, 
                kernel_size=3, stride=step, pad=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    else:
        net.tops[prefix+'/3x3_dw'] = L.ConvolutionDepthwise(net_bottom, num_output=input_channel * time, 
                kernel_size=3, stride=step, pad=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.tops[prefix+'/3x3_dw/bn'] = L.BatchNorm(net.tops[prefix+'/3x3_dw'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
    net.tops[prefix+'/3x3_dw/scale'] = L.Scale(net.tops[prefix+'/3x3_dw/bn'], param=[dict(lr_mult=1, decay_mult=0),
            dict(lr_mult=2, decay_mult=0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops[prefix+'/3x3_dw/relu6'] = L.ReLU6(net.tops[prefix+'/3x3_dw/scale'], in_place=True)    

    net.tops[prefix+'/1x1_down'] = L.Convolution(net.tops[prefix+'/3x3_dw/relu6'], num_output=out, 
                kernel_size=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.tops[prefix+'/1x1_down/bn'] = L.BatchNorm(net.tops[prefix+'/1x1_down'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
    net.tops[prefix+'/1x1_down/scale'] = L.Scale(net.tops[prefix+'/1x1_down/bn'], param=[dict(lr_mult=1, decay_mult=0),
            dict(lr_mult=2, decay_mult=0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)

    if (prefix != 'conv2' and prefix != 'conv6_1' and prefix != 'conv8' and step == 1):
        print ("prefix: ", prefix)
        net.tops[prefix+'/add'] = L.Eltwise(net_bottom, net.tops[prefix+'/1x1_down/scale'])
    else:
        pass

def generate_net(train_lmdb, val_lmdb, train_batch_size, test_batch_size):
    net = caffe.NetSpec()   

    net.data, net.label = L.Data(source=train_lmdb, backend=caffe.params.Data.LMDB, batch_size=train_batch_size, ntop=2, 
	    transform_param=dict(crop_size=224, mean_value= [103.94, 116.78, 123.68]), scale=0.017, include=dict(phase=caffe.TRAIN))
    # note:
    train_data_layer_str = str(net.to_proto())

    net.data, net.label = L.Data(source=val_lmdb, backend=caffe.params.Data.LMDB, batch_size=test_batch_size, ntop=2, 
            transform_param=dict(crop_size=224, mean_value=[103.94, 116.78, 123.68]), scale=0.017, include=dict(phase=caffe.TEST)) 
    # bone
    net.conv1 = L.Convolution(net.data, num_output=32, kernel_size=3, stride=2, pad=1, weight_filler={"type":"xavier"},
            param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)])

    net.tops['conv1/bn'] = L.BatchNorm(net.conv1, param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)

    net.tops['conv1/scale'] = L.Scale(net.tops['conv1/bn'], param=[dict(lr_mult=1, decay_mult=0),
            dict(lr_mult=2, decay_mult=0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)

    net.conv1_relu = L.ReLU6(net.tops['conv1/scale'], in_place=True)

    bottleneck(net, net.conv1_relu, 'conv2', 32, 1, 16, 1)

    bottleneck(net, net.tops['conv2/1x1_down/scale'], 'conv3_1', 16, 6, 24, 2)

    bottleneck(net, net.tops['conv3_1/1x1_down/scale'], 'conv3_2', 24, 6, 24, 1)

    bottleneck(net, net.tops['conv3_2/add'], 'conv4_1', 24, 6, 32, 2)

    bottleneck(net, net.tops['conv4_1/1x1_down/scale'], 'conv4_2', 32, 6, 32, 1)

    bottleneck(net, net.tops['conv4_2/add'], 'conv4_3', 32, 6, 32, 1)

    bottleneck(net, net.tops['conv4_3/add'], 'conv5_1', 32, 6, 64, 2)

    bottleneck(net, net.tops['conv5_1/1x1_down/scale'], 'conv5_2', 64, 6, 64, 1)

    bottleneck(net, net.tops['conv5_2/add'], 'conv5_3', 64, 6, 64, 1)

    bottleneck(net, net.tops['conv5_3/add'], 'conv5_4', 64, 6, 64, 1) 

    bottleneck(net, net.tops['conv5_4/add'], 'conv6_1', 64, 6, 96, 1)

    bottleneck(net, net.tops['conv6_1/1x1_down/scale'], 'conv6_2', 96, 6, 96, 1)    

    bottleneck(net, net.tops['conv6_2/add'], 'conv6_3', 96, 6, 96, 1)

    bottleneck(net, net.tops['conv6_3/add'], 'conv7_1', 96, 6, 160, 2) 

    bottleneck(net, net.tops['conv7_1/1x1_down/scale'], 'conv7_2', 160, 6, 160, 1)

    bottleneck(net, net.tops['conv7_2/add'], 'conv7_3', 160, 6, 160, 1)

    bottleneck(net, net.tops['conv7_3/add'], 'conv8', 160, 6, 320, 1)
    
    net.conv9 = L.Convolution(net.tops['conv8/1x1_down/scale'],
            num_output=1280, kernel_size=1, weight_filler={"type":"xavier"}, param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)])

    net.tops['conv9/bn'] = L.BatchNorm(net.conv9, param=[dict(lr_mult=0, decay_mult=0), 
           dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)

    net.tops['conv9/scale'] = L.Scale(net.tops['conv9/bn'], param=[dict(lr_mult=1, decay_mult=0),
            dict(lr_mult=2, decay_mult=0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.conv9_relu = caffe.layers.ReLU6(net.tops['conv9/scale'], in_place = True)

    # global average pooling
    net.pool10 = L.Pooling(net.conv9_relu, pool=caffe.params.Pooling.AVE, global_pooling=True)

    # 1000 cls
    net.conv11 = L.Convolution(net.pool10, num_output=1000, kernel_size=1, weight_filler={"type":"gaussian","mean":0, "std":0.01},
            param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)])

    # softmax loss
    net.loss = L.SoftmaxWithLoss(net.conv11, net.label, include=dict(phase=caffe.TRAIN))

    # accuracy
    net.accuracy = L.Accuracy(net.conv11, net.label, include=dict(phase=caffe.TEST))
    net.accuracy_top5 = L.Accuracy(net.conv11, net.label, include=dict(phase=caffe.TEST),accuracy_param=dict(top_k=5))
    

    return train_data_layer_str + str(net.to_proto())

def write_net(proto, train_lmdb, val_lmdb):
    with open(proto, 'w') as f:
        f.write(str(generate_net(train_lmdb, val_lmdb, train_batch_size=32, test_batch_size=10)))
        f.close()

if __name__ == '__main__':
    train_lmdb = "/home/raid5/DiskStation/yehao/ilsvrc12_train_lmdb"
    val_lmdb = "/home/raid5/DiskStation/yehao/ilsvrc12_val_lmdb"

    train_proto = "/home/yehao/caffe_SSD/models/tiny-ssd-depthwise/train_val.prototxt"

    deploy_proto = "/home/yehao/caffe_SSD/models/tiny-ssd-depthwise/classifi/deploy.prototxt"

    write_net(train_proto, train_lmdb, val_lmdb)

    #write_net(deploy_proto, train_lmdb, val_lmdb)

    with open(train_proto, 'r') as file:
        data = file.read()

    data = data.replace("conv_dw_param", "convolution_param")

    with open(train_proto, 'w') as file:
        file.write(data)

