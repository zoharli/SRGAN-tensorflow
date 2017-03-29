# SRGAN-tensorflow

A complete tensorflow implementation of SRGAN as proposed in following paper:\
[Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/pdf/1609.04802.pdf)


## Prerequisition:
Before cloning this project,you need to do these things at first:\
  1.install tensorflow(v>=r1.0) in python.\
  2.Get data(images for training and testing) ready.You can either use raw jpeg format data (as default in this project),or use [tfRecord](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) file format which is faster in loading data.\
  3.download [VGG19 npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and put it in the main directory of this project.
  
### Note: 
the vgg19 codes and npy file are adapted from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg).
 
