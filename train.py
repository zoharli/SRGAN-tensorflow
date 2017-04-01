import tensorflow as tf
import vgg19
import srResNet
import discriminator
import time
import random
import os 

learn_rate1=1e-3
learn_rate2=1e-5
batch_size=16
resolution=64
flags='b'+str(batch_size)+'_'+'r'+str(resolution)# 'v' means learning_rate
filenames='r256-512.bin'
srResNet_path='./save/srResNet_b64_r64_v0.01/srResNet4x.ckpt'
log_steps=50
endpoint1=20000
endpoint2=40000

save_path='save/srgan_'+flags
if not os.path.exists(save_path):
    os.mkdir(save_path)

def read(filenames):
    file_names=open(filenames,'rb').read().split('\n')
    random.shuffle(file_names)
    filename_queue=tf.train.string_input_producer(file_names,capacity=3000,num_epochs=100)
    reader=tf.WholeFileReader()
    _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[resolution*4,resolution*4,3])
    random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.train.batch([random_flipped],batch_size,capacity=300)
    rescaled=tf.image.resize_bicubic(minibatch,[resolution,resolution])
    return minibatch,rescaled

 
with tf.device('/cpu:0'):
    minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

dbatch=tf.concat([tf.cast(minibatch,tf.float32),resnet.conv5],0)
vgg=vgg19.Vgg19()
vgg.build(dbatch)
fmap=tf.split(vgg.conv5_4,2)
content_loss=tf.reduce_mean(tf.squared_difference(fmap[0],fmap[1]))

disc=discriminator.Discriminator(dbatch)
D_x,D_G_z=tf.split(tf.squeeze(disc.dense2),2)   
adv_loss=tf.reduce_mean(tf.square(D_G_z-1.0))
gen_loss=(adv_loss+content_loss)
disc_loss=(tf.reduce_mean(tf.square(D_x-1.0)+tf.square(D_G_z)))

global_step=tf.Variable(0,name='global_step')
var_list1=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gen_train_step1=tf.train.AdamOptimizer(learn_rate1).minimize(gen_loss,global_step,var_list1)
gen_train_step2=tf.train.AdamOptimizer(learn_rate2).minimize(gen_loss,global_step,var_list1)
disc_train_step1=tf.train.AdamOptimizer(learn_rate1).minimize(disc_loss,global_step,var_list1)
disc_train_step2=tf.train.AdamOptimizer(learn_rate2).minimize(disc_loss,global_step,var_list1)

with tf.Session() as sess:
    if not os.path.exists(save_path+'srgan.ckpt.meta'):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        loader = tf.train.Saver(var_list=var_list)
        loader.restore(sess,srResNet_path)
        saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess,save_path+'/srgan.ckpt')
        print('saved')
    saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess,save_path+'/srgan.ckpt')
    def save():
        saver.save(sess,save_path+'/srgan.ckpt')
    sess.run(tf.local_variables_initializer())
    step=global_step.eval
    tf.train.start_queue_runners()

    def train(endpoint,gen_step,disc_step):
        while step()<=endpoint:
            if(step()%log_steps==0):
                f=open('train_'+flags+'.txt','a')
                s=time.strftime('%Y-%m-%d %H:%M:%S:',time.localtime(time.time()))+'step '+str(step())+' gen_loss='+str(gen_loss.eval())+' disc_loss='+str(disc_loss.eval())+'\n'
                print(s)
                f.write(s)
                f.close()
                save()
            sess.run(disc_step)
            sess.run(gen_step)
    train(endpoint1,gen_train_step1,disc_train_step1)
    train(endpoint2,gen_train_step2,disc_train_step2)
    print('trainning finished')
