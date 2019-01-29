import sys, os
sys.path.insert(0,'/home/yehao/caffe_SSD/python') # your caffe path

import re
import fileinput

import caffe
from caffe import layers as L

def fire(net, net_bottom, prefix, out1, out2, out3):
    net.tops[prefix+'/squeeze1x1'] = L.Convolution(net_bottom, num_output=out1, 
                kernel_size=1, weight_filler={"type":"xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.tops[prefix+'/squeeze1x1/bn'] = L.BatchNorm(net.tops[prefix+'/squeeze1x1'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops[prefix+'/squeeze1x1/scale'] = L.Scale(net.tops[prefix+'/squeeze1x1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
           
    net.tops[prefix+'/expand1x1'] = L.Convolution(net.tops[prefix+'/squeeze1x1/scale'], num_output=out2, 
                kernel_size=1, weight_filler={"type":"xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.tops[prefix+'/expand1x1/bn'] = L.BatchNorm(net.tops[prefix+'/expand1x1'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops[prefix+'/expand1x1/scale'] = L.Scale(net.tops[prefix+'/expand1x1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
        
    net.tops[prefix+'/relu_expand1x1'] = L.ReLU(net.tops[prefix+'/expand1x1/scale'], in_place=True)

    net.tops[prefix+'/expand3x3'] = L.Convolution(net.tops[prefix+'/squeeze1x1/scale'], num_output=out3, pad=1, stride=1,
                kernel_size=3, weight_filler={'type':'xavier'}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.tops[prefix+'/expand3x3/bn'] = L.BatchNorm(net.tops[prefix+'/expand3x3'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops[prefix+'/expand3x3/scale'] = L.Scale(net.tops[prefix+'/expand3x3/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
            
    net.tops[prefix+'/relu_expand3x3'] = L.ReLU(net.tops[prefix+'/expand3x3/scale'], in_place=True)
    
    net.tops[prefix+'/concat'] = L.Concat(net.tops[prefix+'/relu_expand1x1'], net.tops[prefix+'/relu_expand3x3'])

    return net.tops[prefix+'/concat'] 

def bottleneck(net, net_bottom, prefix, input_channel, time, out, step):
    net.tops[prefix+'/1x1_up'] = L.Convolution(net_bottom, num_output=input_channel * time, 
                kernel_size=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.tops[prefix+'/1x1_up/bn'] = L.BatchNorm(net.tops[prefix+'/1x1_up'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
    net.tops[prefix+'/1x1_up/scale'] = L.Scale(net.tops[prefix+'/1x1_up/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops[prefix+'/1x1_up/relu6'] = L.ReLU6(net.tops[prefix+'/1x1_up/scale'], in_place=True)

    net.tops[prefix+'/3x3_dw'] = L.ConvolutionDepthwise(net.tops[prefix+'/1x1_up/relu6'], num_output=input_channel * time, 
                kernel_size=3, stride=step, pad=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.tops[prefix+'/3x3_dw/bn'] = L.BatchNorm(net.tops[prefix+'/3x3_dw'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
    net.tops[prefix+'/3x3_dw/scale'] = L.Scale(net.tops[prefix+'/3x3_dw/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops[prefix+'/3x3_dw/relu6'] = L.ReLU6(net.tops[prefix+'/3x3_dw/scale'], in_place=True)    

    net.tops[prefix+'/1x1_down'] = L.Convolution(net.tops[prefix+'/3x3_dw/relu6'], num_output=out, 
                kernel_size=1, weight_filler={"type": "xavier"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.tops[prefix+'/1x1_down/bn'] = L.BatchNorm(net.tops[prefix+'/1x1_down'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)
    net.tops[prefix+'/1x1_down/scale'] = L.Scale(net.tops[prefix+'/1x1_down/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)

    if (step == 1 and prefix != 'conv2' and prefix != 'conv6_1' and prefix != 'conv8'):
        net.tops[prefix+'/add'] = L.Eltwise(net_bottom, net.tops[prefix+'/1x1_down/scale'])
        #return net.tops[prefix+'/add']
    else:
        pass
        #return net.tops[prefix+'/1x1_down/scale']

    

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

    net.tops['conv1/scale'] = L.Scale(net.tops['conv1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)

    net.conv1_relu = L.ReLU6(net.tops['conv1/scale'], in_place=True)
    """
    net.tops['conv2/1x1_down/scale'] = bottleneck(net, net.conv1_relu, 'conv2', 32, 1, 16, 1)

    net.tops['conv3_1/1x1_down/scale'] = bottleneck(net, net.tops['conv2/1x1_down/scale'], 'conv3_1', 16, 2, 24, 2)

    net.tops['conv3_2/1x1_down/scale'] = bottleneck(net, net.tops['conv3_1/1x1_down/scale'], 'conv3_2', 24, 2, 24, 1)

    net.tops['conv4_1/1x1_down/scale'] = bottleneck(net, net.tops['conv3_2/1x1_down/scale'], 'conv4_1', 24, 2, 32, 2)

    net.tops['conv4_2/1x1_down/scale'] = bottleneck(net, net.tops['conv4_1/1x1_down/scale'], 'conv4_2', 32, 2, 32, 1)

    #net.tops['conv4_3/bottleneck_1'] = bottleneck(net, net.tops['conv4_2/bottleneck_1'], 'conv4_3', 32, 2, 32, 1)

    net.tops['conv5_1/1x1_down/scale'] = bottleneck(net, net.tops['conv4_2/1x1_down/scale'], 'conv5_1', 32, 2, 64, 2)

    net.tops['conv5_2/1x1_down/scale'] = bottleneck(net, net.tops['conv5_1/1x1_down/scale'], 'conv5_2', 64, 2, 64, 1)

    net.tops['conv5_3/1x1_down/scale'] = bottleneck(net, net.tops['conv5_2/1x1_down/scale'], 'conv5_3', 64, 2, 64, 1)

    #net.tops['conv5_4/bottleneck_1'] = bottleneck(net, net.tops['conv5_3/bottleneck_1'], 'conv5_4', 64, 2, 64, 1)    

    net.tops['conv6_1/1x1_down/scale'] = bottleneck(net, net.tops['conv5_3/1x1_down/scale'], 'conv6_1', 64, 2, 96, 1)

    net.tops['conv6_2/1x1_down/scale'] = bottleneck(net, net.tops['conv6_1/1x1_down/scale'], 'conv6_2', 96, 2, 96, 1)    

    #net.tops['conv6_3/bottleneck_1'] = bottleneck(net, net.tops['conv6_2/bottleneck_1'], 'conv6_3', 96, 2, 96, 1)

    net.tops['conv7_1/1x1_down/scale'] = bottleneck(net, net.tops['conv6_2/1x1_down/scale'], 'conv6_3', 96, 2, 160, 2) 

    net.tops['conv7_2/1x1_down/scale'] = bottleneck(net, net.tops['conv7_1/1x1_down/scale'], 'conv7_2', 160, 2, 160, 1)

    #net.tops['conv7_3/bottleneck_1'] = bottleneck(net, net.tops['conv7_2/bottleneck_1'], 'conv7_3', 160, 2, 160, 1) 

    #net.tops['conv8/bottleneck_1'] = bottleneck(net, net.tops['conv7_3/bottleneck_1'], 'conv8', 160, 2, 320, 1)
	"""

    bottleneck(net, net.conv1_relu, 'conv2', 32, 1, 16, 1)

    bottleneck(net, net.tops['conv2/1x1_down/scale'], 'conv3_1', 16, 2, 24, 2)

    bottleneck(net, net.tops['conv3_1/1x1_down/scale'], 'conv3_2', 24, 2, 24, 1)

    bottleneck(net, net.tops['conv3_2/add'], 'conv4_1', 24, 2, 32, 2)

    bottleneck(net, net.tops['conv4_1/1x1_down/scale'], 'conv4_2', 32, 2, 32, 1)

    
    #net.tops['conv4_3/bottleneck_1'] = bottleneck(net, net.tops['conv4_2/bottleneck_1'], 'conv4_3', 32, 2, 32, 1)

    bottleneck(net, net.tops['conv4_2/add'], 'conv5_1', 32, 2, 64, 2)


    bottleneck(net, net.tops['conv5_1/1x1_down/scale'], 'conv5_2', 64, 2, 64, 1)


    bottleneck(net, net.tops['conv5_2/add'], 'conv5_3', 64, 2, 64, 1)

    #net.tops['conv5_4/bottleneck_1'] = bottleneck(net, net.tops['conv5_3/bottleneck_1'], 'conv5_4', 64, 2, 64, 1)    

    bottleneck(net, net.tops['conv5_3/add'], 'conv6_1', 64, 2, 96, 1)

    bottleneck(net, net.tops['conv6_1/1x1_down/scale'], 'conv6_2', 96, 2, 96, 1)    

    #net.tops['conv6_3/bottleneck_1'] = bottleneck(net, net.tops['conv6_2/bottleneck_1'], 'conv6_3', 96, 2, 96, 1)

    bottleneck(net, net.tops['conv6_2/add'], 'conv6_3', 96, 2, 160, 2) 

    bottleneck(net, net.tops['conv6_3/1x1_down/scale'], 'conv7_1', 160, 2, 160, 1)
    
    #net.tops['conv7_3/bottleneck_1'] = bottleneck(net, net.tops['conv7_2/bottleneck_1'], 'conv7_3', 160, 2, 160, 1) 

    #net.tops['conv8/bottleneck_1'] = bottleneck(net, net.tops['conv7_3/bottleneck_1'], 'conv8', 160, 2, 320, 1)

    net.conv8 = caffe.layers.Convolution(net.tops['conv7_1/add'],
            num_output=1000, kernel_size=1, weight_filler={"type":"gaussian","mean":0, "std":0.01},
            param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)])

    net.tops['conv8/bn'] = caffe.layers.BatchNorm(net.conv8, param=[dict(lr_mult=0, decay_mult=0), 
           dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=False)

    net.tops['conv8/scale'] = caffe.layers.Scale(net.tops['conv8/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.conv8_relu = caffe.layers.ReLU6(net.tops['conv8/scale'], in_place = True)

    # global average pooling
    net.pool9 = L.Pooling(net.conv8_relu, pool=caffe.params.Pooling.AVE, global_pooling=True)

    # softmax loss
    net.loss = L.SoftmaxWithLoss(net.pool9, net.label, include=dict(phase=caffe.TRAIN))

    # accuracy
    net.accuracy = L.Accuracy(net.pool9, net.label, include=dict(phase=caffe.TEST))
    net.accuracy_top5 = L.Accuracy(net.pool9, net.label, include=dict(phase=caffe.TEST),accuracy_param=dict(top_k=5))
    

    return train_data_layer_str + str(net.to_proto())

def write_net(proto, train_lmdb, val_lmdb):
    with open(proto, 'w') as f:
        f.write(str(generate_net(train_lmdb, val_lmdb, train_batch_size=64, test_batch_size=25)))
        f.close()

if __name__ == '__main__':
    train_lmdb = "/media/yehao/disk0/ILSVRC2012/ilsvrc12_train_lmdb"
    val_lmdb = "/media/yehao/disk0/ILSVRC2012/ilsvrc12_val_lmdb"

    train_proto = "/home/yehao/caffe_SSD/models/tiny-ssd-depthwise/train_val.prototxt"
    deploy_proto = "/home/yehao/caffe_SSD/models/tiny-ssd-depthwise/classifi/deploy.prototxt"

    write_net(train_proto, train_lmdb, val_lmdb)

    #write_net(deploy_proto, train_lmdb, val_lmdb)

    with open(train_proto, 'r') as file:
	data = file.read()

    data = data.replace("conv_dw_param", "convolution_param")

    with open(train_proto, 'w') as file:
	file.write(data)

