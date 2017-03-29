import tensorflow as tf
import srResNet
import time
import random
import os 

learn_rate=1e-3 #recommended
batch_size=64 #recommended
resolution=64 #downsampled image resolution, in this case 64x64
flags='b'+str(batch_size)+'_r'+str(resolution)+'_v'+str(learn_rate)#set for practicers to try different setups 
filenames='r256-512.bin' #put images' paths to this file,one image path for each row,e.g. ./data/123.JPEG, or define another way of loading images in read()
total_steps=int(2e5+1)
log_steps=50 #interval to save the model parameters

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
    minibatch=tf.train.batch([random_flipped],batch_size,capacity=500)
    rescaled=tf.image.resize_bicubic(minibatch,[resolution,resolution])
    return minibatch,rescaled

with tf.device('/cpu:0'):
    minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
mse=tf.reduce_mean(tf.squared_difference(tf.cast(minibatch,tf.float32),resnet.conv5))
train_step=tf.train.AdamOptimizer(learn_rate).minimize(mse)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    saver = tf.train.Saver()
    def save():
        saver.save(sess,save_path+'/srResNet4x.ckpt')
    for i in xrange(total_steps):
        if(i%log_steps):
            sess.run(train_step)
        else:    
            s=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+':step '+str(i)+' loss '+str(mse.eval())
            print(s)
            f=open('pretrain_'+flags+'.txt','a')
            f.write(s+'\n')
            f.close()
            save()

print('done')
