import caffe
from caffe import layers as L, params as P

def conv_relu(data, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(data, kernel_size=ks, stride=stride, num_output=nout,
                         pad=pad, weight_filler=dict(type='xavier'),bias_filler=dict(type='constant', value=0),
                         param=[dict(lr_mult=1), dict(lr_mult=2)])

    return conv, L.ReLU(conv, in_place=True)

def max_pool(data, ks=2, stride=2):
    return L.Pooling(data, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def u_net(split):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(41.4661, 69.1061, 126.993),
            seed=1337)
    if split == 'train':
        pydata_params['train_dir'] = '../image_augmentor/DRIVE/training'
        pylayer = 'TRAINSegDataLayer'
    else:
        pydata_params['val_dir'] = '../image_augmentor/DRIVE/val'
        pylayer = 'VALSegDataLayer'
    n.data, n.label = L.Python(module='train_val', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # layer group 1
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 32)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 32)
    n.pool1 = max_pool(n.relu1_2)

    # layer group 2
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 64)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 64)
    n.pool2 = max_pool(n.relu2_2)

    # layer group 3
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 128)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 128)

    # layer group 4
    n.upconv4 = L.Deconvolution(n.relu3_2, param=[dict(lr_mult=1), dict(lr_mult=2)],
                                convolution_param=dict(num_output=128, kernel_size=2, stride=2))
    n.concat4 = L.Concat(n.upconv4, n.relu2_2, axis=1)
    n.conv4_1, n.relu4_1 = conv_relu(n.concat4, 64)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 64)

    # layer group 5
    n.upconv5 = L.Deconvolution(n.relu4_2, param=[dict(lr_mult=1), dict(lr_mult=2)],
                                convolution_param=dict(num_output=64, kernel_size=2, stride=2))
    n.concat5 = L.Concat(n.upconv5, n.conv1_2, axis=1)
    n.conv5_1, n.relu5_1 = conv_relu(n.concat5, 32)
    # n.drop5 = L.Dropout(n.relu5_1, dropout_ratio=0.2, in_place=True)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 32)

    # layer group 6
    n.score = L.Convolution(n.relu5_2, pad=0, kernel_size=1, num_output=2,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.seg = L.Dropout(n.score, dropout_ratio=0.5, in_place=True)

    # others
    n.loss = L.SoftmaxWithLoss(n.seg, n.label,
                               loss_param=dict(normalize=False))
    # n.softmax = L.Softmax(n.seg,
    #                       include={'phase':caffe.TEST})
    # n.argmax = L.ArgMax(n.softmax, axis=1,
    #                     include={'phase':caffe.TEST})
    # n.accuracy = L.Accuracy(n.seg, n.label, exclude={'stage': 'deploy'})

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(u_net('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(u_net('val')))

# show all avaliable caffe layer
def print_layers():
    layer_lsit = caffe.layer_type_list()
    for layer in layer_lsit:
        print layer

if __name__ == '__main__':
    make_net()


