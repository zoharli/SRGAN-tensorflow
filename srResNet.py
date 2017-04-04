import tensorflow as tf
from utils import *
class srResNet:
    def __init__(self,T,train_mode=1,name='srResNet'):
        with tf.variable_scope(name):
            self.train_mode=train_mode
            conv1=conv_layer(T,[5,5,3,64],1)
            relu1=leaky_relu(conv1)
            block=[]
            for i in xrange(16):
                block.append(self.residual_block(block[-1] if i else relu1))
            conv2=conv_layer(block[-1],[3,3,64,64],1)
            bn1=batch_norm(conv2) if self.train_mode else conv2
            sum1=tf.add(bn1,relu1)
            conv3=conv_layer(sum1,[3,3,64,256],1)
            ps1=tf.depth_to_space(conv3,2) #pixel-shuffle
            relu2=leaky_relu(ps1)
            conv4=conv_layer(relu2,[3,3,64,256],1)
            ps2=tf.depth_to_space(conv4,2)
            relu3=leaky_relu(ps2)
            self.conv5=conv_layer(relu3,[3,3,64,3],1)
    def residual_block(self,T,name='block'):
        with tf.variable_scope(name):
            conv1=conv_layer(T,[3,3,64,64],1)
            bn1=batch_norm(conv1) if self.train_mode else conv1
            relu=leaky_relu(bn1)
            conv2=conv_layer(relu,[3,3,64,64],1)
            bn2=batch_norm(conv2) if self.train_mode else conv2
            return tf.add(T,bn2)
