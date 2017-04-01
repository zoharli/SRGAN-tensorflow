# SRGAN-tensorflow

A complete tensorflow implementation of SRGAN as proposed in following paper:\
[Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/pdf/1609.04802.pdf)


## Dependencies:
  `tensorflow version>=r1.0`
 
## Usage:
  1. Run`$ git clone https://github.com/zoharli/SRGAN-tensorflow`
  2. Get data(images for training and testing) ready,the way of loading image data is specified in `pretrain.py`.You can either use raw jpeg format data (as default in this project),or use [tfRecord](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) file format which is faster in loading data.
  3. Download [VGG19 npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and put it in the main directory of this project.
  4. Run`$ python pretrain.py`
  5. Run`$ python train.py`

### Note:
1. You can change the batch size, iterating steps and learning rate to adapt your computing resources.The default setup in code is run under system with a single TELSA K80 GPU(for training) and multi-cpu(for loading and preprocessing),and it takes about 6 minutes for every 100 steps of pretraining and 8 mins for formal training.
2. Testing code and trained parameter files are temporarily unavailable. 
3. The vgg19 codes and npy file are adapted from [machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg).
4. I used [least square loss](https://pdfs.semanticscholar.org/0bbc/35bdbd643fb520ce349bdd486ef2c490f1fc.pdf) to substitute the original cross entropy loss function as adversarial loss.The decent weight leverage between content loss and adversarial loss(least suqare loss) are temporarily not provided.You can either try different weights or just use the cross entropy loss and set the weights as presented in the original paper.


#### update 20/3/17：
 * Psnr and ssim assessment are added.
 * Automatic saving and loading are enabled.
