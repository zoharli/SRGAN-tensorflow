import tensorflow as tf
import numpy as np

def conv_layer(input,filter_shape,stride,name='conv'):
    with tf.variable_scope(name):
        conv=tf.nn.conv2d(input,conv_filter(filter_shape),strides=[1,stride,stride,1],padding='SAME')
        bias=tf.Variable(tf.zeros(conv.shape.as_list()[3]),name='bias')
        return tf.nn.bias_add(conv,bias)
def batch_norm(input,name='BN'):
    ch_num=input.shape.as_list()[3]
    with tf.variable_scope(name):
        return tf.nn.fused_batch_norm(input,
                                  offset=tf.Variable(tf.zeros(ch_num),name='offset'),
                                  scale=tf.Variable(tf.ones(ch_num),name='scale'),
                                  name=name)[0]
def conv_filter(shape,name='filter'):
    #initialize weights as the way proposed in [He et.cl.:Delving_Deep_into_rectifiers_ICCV_2015_paper]
    return tf.Variable(tf.random_normal(shape,stddev=np.sqrt(2.0/shape[0]/shape[1]/shape[2])),name=name)
def leaky_relu(x,alpha=0.1,name='lrelu'):
     with tf.name_scope(name):
         x=tf.maximum(x,alpha*x)
         return x
def batch_mse_psnr(dbatch):
    im1,im2=np.split(dbatch,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return np.mean(mse),psnr
def batch_ssim(dbatch):
    im1,im2=np.split(dbatch,2)
    imgsize=im1.shape[1]*im1.shape[2]
    avg1=im1.mean((1,2),keepdims=1)
    avg2=im2.mean((1,2),keepdims=1)
    std1=im1.std((1,2),ddof=1)
    std2=im2.std((1,2),ddof=1)
    cov=((im1-avg1)*(im2-avg2)).mean((1,2))*imgsize/(imgsize-1)
    avg1=np.squeeze(avg1)
    avg2=np.squeeze(avg2)
    k1=0.01
    k2=0.03
    c1=(k1*255)**2
    c2=(k2*255)**2
    c3=c2/2
    return np.mean((2*avg1*avg2+c1)*2*(cov+c3)/(avg1**2+avg2**2+c1)/(std1**2+std2**2+c2))
