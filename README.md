# SRGAN-tensorflow

A complete tensorflow implementation of SRGAN as proposed in following paper:\
[Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/pdf/1609.04802.pdf)


## Dependencies:
  `tensorflow version>=r1.0`
 
## How to run this project:
  1. run`$ git clone https://github.com/zoharlee/SRGAN-tensorflow`\
  2. Get data(images for training and testing) ready,the way of loading image data is specified in `pretrain.py`.You can either use raw jpeg format data (as default in this project),or use [tfRecord](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) file format which is faster in loading data.\
  3. download [VGG19 npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and put it in the main directory of this project.\
  4. run`$ python pretrain.py`\
  5. run`$ python train.py`

### Note:
1.You can change the batch size, iterating steps and learning rate to adapt your computing resources.The default setup in code is run under system with a single TELSA K80 GPU and multi-cpu,and it takes about 6 minutes for every 100 steps of pretraining and 8 mins for formal training.\
2.testing code and trained parameter files are temporarily unavailable. \
3.the vgg19 codes and npy file are adapted from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg).
