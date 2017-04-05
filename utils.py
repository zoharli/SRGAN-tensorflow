import tensorflow as tf
import numpy as np

def conv_layer(inputs,ochannels,ksize,stride=1,activation=None,name='conv'):
    return tf.layers.conv2d(inputs,ochannels,ksize,stride,'same',activation=activation,kernel_initializer=tf.random_normal_initializer(0,np.sqrt(2.0/ksize/ksize/(inputs.shape.as_list()[3]+ochannels)*2)),name=name)#msra initialization
def batch_norm(inputs,training=True):
    return tf.layers.batch_normalization(inputs,training=training)
def leaky_relu(x,alpha=0.1,name='lrelu'):
    return tf.maximum(x,alpha*x,name=name)
def batch_mse_psnr(dbatch):
    dbatch=dbatch[:,4:-4,4:-4,:]
    im1,im2=np.split(dbatch,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return np.mean(mse),psnr
def batch_y_psnr(dbatch):
    dbatch=dbatch[:,4:-4,4:-4,:]
    r,g,b=np.split(dbatch,3,axis=3)
    y=16+np.squeeze(0.183*r+0.614*g+0.06*b)
    im1,im2=np.split(y,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return psnr
def batch_ssim(dbatch):
    dbatch=dbatch[:,4:-4,4:-4,:]
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
