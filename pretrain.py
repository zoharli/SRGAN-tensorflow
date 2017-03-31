import tensorflow as tf
import srResNet
import time
import random
import os 
from utils import *

learn_rate=0.01 #recommended
batch_size=64 #recommended
resolution=64 #downsampled image resolution, in this case 64x64
flags='b'+str(batch_size)+'_r'+str(resolution)+'_v'+str(learn_rate)#set for practicers to try different setups 
filenames='r256-512.bin' #put images' paths to this file,one image path for each row,e.g. ./data/123.JPEG, or define another way of loading images in read()
total_steps=int(2e5)
log_steps=100 #interval to save the model parameters

if not os.path.exists('save'):
    os.mkdir('save')

save_path='save/srResNet_'+flags
if not os.path.exists(save_path):
    os.mkdir(save_path)

def read(filenames):
    file_names=open(filenames,'rb').read().split('\n')
    random.shuffle(file_names)
    filename_queue=tf.train.string_input_producer(file_names,capacity=5000,num_epochs=100)#shuffled input_producer by default
    reader=tf.WholeFileReader()
    _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[resolution*4,resolution*4,3])
    random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.cast(tf.train.batch([random_flipped],batch_size,capacity=500),tf.float32)
    rescaled=tf.image.resize_bicubic(minibatch,[resolution,resolution])
    return minibatch,rescaled

with tf.device('/cpu:0'):
    minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
MSE=tf.reduce_mean(tf.squared_difference(minibatch,resnet.conv5))
global_step=tf.Variable(0,name='global_step')
train_step=tf.train.AdamOptimizer(learn_rate).minimize(MSE,global_step)
dbatch=tf.concat([minibatch,resnet.conv5],0)

with tf.Session() as sess:
    if not os.path.exists(save_path+'/srResNet.ckpt.meta'):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.save(sess,save_path+'/srResNet.ckpt')
    saver=tf.train.Saver()
    saver.restore(sess,save_path+'/srResNet.ckpt')
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    def save():
        saver.save(sess,save_path+'/srResNet4x.ckpt')
    step=global_step.eval
    while step()<=total_steps:
        if(step()%log_steps==0):
            d_batch=dbatch.eval()
            mse,psnr=batch_mse_psnr(d_batch)
            ssim=batch_ssim(d_batch)
            s=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+':step '+str(step())+' mse:'+str(mse)+' psnr:'+str(psnr)+' ssim:'+str(ssim)  
            print(s)
            f=open('pretrain_'+flags+'.txt','a')
            f.write(s+'\n')
            f.close()
            save()
        sess.run(train_step)

print('done')
