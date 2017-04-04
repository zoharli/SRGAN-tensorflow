import tensorflow as tf
import srResNet
import time
import random
import os 
from utils import *
import vgg19

learn_rate=0.001
batch_size=32 #recommended
resolution=64 #downsampled image resolution, in this case 64x64
flags='b'+str(batch_size)+'_r'+str(resolution)+'_v'+str(learn_rate)+'_leaky_tanh'#set for practicers to try different setups 
filenames='r256-512.bin' #put images' paths to this file,one image path for each row,e.g. ./data/123.JPEG, or define another way of loading images in read()
total_steps=int(10e5)
log_steps=50 #interval to save the model parameters

if not os.path.exists('save'):
    os.mkdir('save')

save_path='save/srResNet_'+flags
if not os.path.exists(save_path):
    os.mkdir(save_path)

def read(filenames):
    file_names=open(filenames,'rb').read().split('\n')
    random.shuffle(file_names)
    filename_queue=tf.train.string_input_producer(file_names,capacity=3000,num_epochs=100)#shuffled input_producer by default
    reader=tf.WholeFileReader()
    _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[resolution*4,resolution*4,3])
    random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.cast(tf.train.batch([random_flipped],batch_size,capacity=300),tf.float32)
    rescaled=tf.image.resize_bicubic(minibatch,[resolution,resolution])
    rescaled=rescaled*2/255-1
    return minibatch,rescaled

with tf.device('/cpu:0'):
    minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
result=(tf.tanh(resnet.conv5)+1)*255/2
dbatch=tf.concat([minibatch,result],0)
MSE=tf.losses.mean_squared_error(minibatch,result)
global_step=tf.Variable(0,name='global_step')
train_step=tf.train.AdamOptimizer(learn_rate).minimize(MSE,global_step)

with tf.Session() as sess:
    if not os.path.exists(save_path+'/srResNet.ckpt.meta'):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess,save_path+'/srResNet.ckpt')
    saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess,save_path+'/srResNet.ckpt')
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    def save():
        saver.save(sess,save_path+'/srResNet.ckpt')
    step=global_step.eval
    while step()<=total_steps:
        if(step()%log_steps==0):
            d_batch=dbatch.eval()
            mse,psnr=batch_mse_psnr(d_batch)
            ypsnr=batch_y_psnr(d_batch)
            ssim=batch_ssim(d_batch)
            s=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+':step '+str(step())+' mse:'+str(mse)+' psnr:'+str(psnr)+' ssim:'+str(ssim)+' y_psnr='+str(ypsnr)  
            print(s)
            f=open('info.pretrain_'+flags,'a')
            f.write(s+'\n')
            f.close()
            save()
        sess.run(train_step)

print('done')
