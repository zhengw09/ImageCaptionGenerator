# Image Caption Generator with Scalable Training Method
A repository for practice of Tensoflow

## References
<br>[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)</br>
<br>[yunjey](https://github.com/yunjey)'s implementation: https://github.com/yunjey/show-attend-and-tell<br>


## Prerequisites

Please clone this repo and [pycocoevalcap](https://github.com/tylin/coco-caption.git) in same directory.
This is the library for BLEU score computation.

Python version: Python 2.7
Tensorflow version: 1.5 (with GPU version)   
Required Python libraries: numpy, scipy, scikit-image, hickle, Pillow

Data set: 
The images are resized to 224x224. 
The data set is divided into training set, develop set and test set in './image/train_resized', './image/dev_resized' and './image/test_resized'. The corresponding file names for training, dev and test set are recorded in './data/Flickr_8k.train.txt', './data/Flickr_8k.dev.txt' and './data/Flickr_8k.test.txt', respectively.
The ground-truth captions for Flickr8k images are in './data/Flickr8k.token.txt'.
The pre-trained Vggnet model is in './data/imagenet-vgg-verydeep-19.mat'.
The additional training cat images are in './image/cat_resized'. The test images for cat/dog image testing is in './image/animal_test'. The captions for training cat images are in './data/cat_annotations.txt'.

Directory dependencies:
Please make sure './data', './model/lstm', './image', './core' directories all exist.
Please make sure './data/annotations', './data/train', './data/dev' and './data/test' exist, which are directories for intermediate data storage.
Please make sure 'imagenet-vgg-verydeep-19.mat' in './data/'


## Train the model

To train the image caption generating model, run the command below.
```bash
$ python train.py
```
The model will be saved to './model/lstm'.


## Evaluate the model 

After training we can evaluate the model generated.
To get the results on the test data set, run command below. 
```bash
$ python test.py
```

To get the results on the cat/dog test data set, please copy all files in './image/animal_test' to './image/pred' and run command below.
```bash
$ python animal_predict.py
```

To generate captions for new images, please upload images to be captioned in './image/pred' and run command below.
```bash
$ python predict.py
```
The software will resize and move all images to './image/pred_resized' and generate the predict resuls in 'output.txt'. 