# TangrIAm Project

The project is partnership between Exploradôme museum, OCTO Technology and Microsoft and it aims to introduce the concept and application of artificial intelligence to young children. The specific application developed for the project is to apply object detection to live tangram solving.

A tangram is a dissection puzzle consisting of seven flat polygons (5 triangles, 1 square and 1 parallelogram) which are combined to obtain a specific shape. The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap.

Within the framework of the project, 12 tangram selected shapes act as classes for the object detector:

![image](https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS)

* boat(bateau), 
* bowl(bol), 
* cat(chat), 
* heart(coeur), 
* swan(cygne), 
* rabbit(lapin), 
* house(maison), 
* hammer(marteau), 
* mountain(montagne), 
* bridge(pont), 
* fox(renard), 
* turtle(tortue)

## Objective

Classify tangram shapes from a live video stream using transfer learning as the main basis of our model.

## Table Of Contents
-  [Dataset Creation](#Dataset-Creation)
   * [1. Video recording](#1-Video-recording)
   * [2. From video to images](#2-From-video-to-images)
   * [3. Images labeling](#3-Images-labeling)
   * [4. Initial Dataset](#4-Initial-Dataset)
   * [5. Data augmentation](#5-Data-augmentation)
-  [Model Creation](#Model-Creation)
   * [Transfer learning](#Transfer-learning)
   * [Apply Transfer Learning](#Apply-Transfer-Learning)
   * [Results or improvement strategy](#Results-or-improvement-strategy)
-  [Getting Started](#Getting-Started)
   * [Folders](#Folders)
   * [Installation and Usage](#Installation-and-Usage)
   * [Get more models](#Get-more-models)
   * [Inference](#Inference)
-  [Team](#Team)

# Dataset Creation

## 1. Video recording
The dataset used to train the models was created by recording video stream of people solving a tangram puzzle and capturing the frames which include completed tangram shapes. The respective frames were used to compile an image dataset with images for each class/tangram shape.

## 2. From video to images

The video recording was processed to create an image dataset as follows:
  * Extract 1 frame/second
  * Half each frame to obtain more images
  * Manually select images with no obstruction tangram shapes (e.g. human hands on the tangram surface)

## 3. Image labeling
Transfer learning with TensorFlow requires an input dataset with a directory structure as below (ordered images in the respective category folder):
```
├──  multilabel_data  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├── 
```
A download link (and instructions) are available in the GitHub repository.

## 4. Initial Dataset

Some categories are more present than other in the initial dataset.

| Label           |  Total images | 
|-----------------|------|
|boat(bateau)     | 716  | 
| bowl(bol)       | 248  |  
| cat(chat)       | 266  | 
| heart(coeur)    | 273  |  
| swan(cygne)     | 321  |  
| rabbit(lapin)   | 257  |  
| house(maison)   | 456  |  
| hammer(marteau) | 403  |  
| mountain(montagne)  |  573 |  
| bridge(pont)    | 709  |  
| fox(renard)     | 768  |  
| turtle(tortue)  | 314  |  
| TOTAL           | 5304 | 

## 5. Data augmentation
Data augmentation is a strategy to increase the diversity of data available for training models, without actually collecting new data.
For this project different types of image augmentations were applied to the initial dataset to create more images and increase variability. 

Data Augmentation with python scripts:
  * Contrast changes (1.5 #brightens the image) with PIL and ImageEnhance with `Brightness()`
  * Blurring (applied after contrast change) with OpenCV and cv2 with `gaussianblur()`

`ImageDataGenerator` with TensorFlow:
  * Rescaling: 1./255 is to transform every pietxel value from range [0,255] -> [0,1]
  * Rotation: each picture is rotated with a random angle from 0° to 90°
  * Flipping: each picture gets flipped on both axis (vertical and horizontal)
  * Split train_full or train_balanced dataset to train and validation dataset (= 30% of train dataset)

```python
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True,
                    validation_split=0.3)
```

| Label           |  Before Data Augmentation  |   After Data Augmentation* | 
|-----------------|---------------|----------------|
| boat(bateau)    | 716           |   2148         | 
| bowl(bol)       | 248           |   744          | 
| cat(chat)       | 266           |   800          | 
| heart(coeur)    | 273           |   820          | 
| swan(cygne)     | 321           |   964          | 
| rabbit(lapin)   | 257           |   772          | 
| house(maison)   | 456           |   1368         | 
| hammer(marteau) | 403           |   1209         | 
| mountain(montagne)  |  573      |   1720         | 
| bridge(pont)    | 709           |   2128         | 
| fox(renard)     | 768           |   2304         |  
| turtle(tortue)  | 314           |   942          |  
| **TOTAL**           | **5304**          |   **15919**        | 

* with python script :
A next step is to balance the datasets and keep, for each class, assigned randomly:
  - 400 images for the training dataset
  - 80 images (20% of 400) for the test dataset
  
The final dataset has the following directory structure:
```
├──  train  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├──  test  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│   
└── 
```

To access the dataset, visit the [Google Drive link](https://drive.google.com/drive/folders/1LQO_zfVZ-niiVsCqzQEUEZHry8aATK2s?usp=sharing). The folder contains both training and validation sets.

# Model Creation
## Transfer learning
**What is Transfer Learning?**
Transfer learning is a machine learning technique in which a network that has already been trained to perform a specific task is repurposed as a starting point for another similar task. 

**Transfer Learning Strategies & Advantages:**
The transfer learning strategy adopted for the project is to:
   * Initialize the CNN network with the pre-trained weights
   * Retrain the entire CNN network while setting the learning rate to be very small, which avoids drastic changes in the pre-trained weights
   
The advantage of transfer learning is that it provides fast training progress since we're not starting from scratch. Transfer learning is also very useful when you have a small training dataset available.

**Using a Pretrained Model:**
There are two ways to create models with Keras API. Here, the sequential model is used. The sequential model is a linear stack of layers. 
You can simply keep adding layers in a sequential model just by calling “add” method. 

The two pretrained models tested on the project dataset are:
* [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2): lightweight, used for laptops
* [InceptionV3 + L2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3): heavier, used for image analysis

**Transfer Learning with Image Data:**
It is common to perform transfer learning with predictive modeling problems that use image data as input.

This may be a prediction task that takes photographs or video data as input.

For these type of tasks, it is common to use a deep learning model pre-trained for a large and challenging image classification task such as the [ImageNet](http://www.image-net.org/) 1000-class photograph classification competition.

These models can be downloaded and incorporated directly into new models that expect image data as input.

## Apply Transfer Learning

**Inception V3**
```python
inception = InceptionV3(weights='imagenet', include_top=False)
```

**MobileNetV2**
```python
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

For both, we added:
```python    
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()    

prediction_layer = tf.keras.layers.Dense(12, activation='softmax')    

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```

## Results or improvement strategy
**MobileNetV2**

![MobileNetV2-image1](https://drive.google.com/uc?export=view&id=1_RqCPHx_AbemYZ_75gaO65Ou1kaoC-fl)

![MobileNetV2-image2](https://drive.google.com/uc?export=view&id=1L4NB8zh11bYCjVsDDtYGixlVvx9-AUPk)

The MobileNetV2 has a good accuracy overall on pictures never shown to the model : overall accuracy is 87%. It can also work quickly at a speed of 5 frames per second, more than enough for the project.

Given the following classification report and confusion matrix, the most issues are found between maison (house) and marteau (hammer), cygne (swan) and torue (turtle), and bol (bowl) and marteau (hammer) .

More training on these classes is required to achieve a more homogeneous accuracy on each classes. Some classes perform perfectly and other less, which gives overall a good accuracy but with uncertainties on some classes.

**InceptionV3**

![IV3-cm](https://drive.google.com/uc?export=view&id=13AZjhMdloIeWmDrRvZWygK62T-b0PIbQ)

![InceptionV3-image2](https://drive.google.com/uc?export=view&id=16JgUAb4uHdvIc6bEEd_qwei9Ou7cySZf)

Using InceptionV3 with 30 epochs , we were able to achieve an accuracy of 98% over our test dataset.
By looking at the confusion matrix,  The classifier has a little trouble with 5 kinds of images. It has a little trouble with Muntain, Rabbit, Swan, cat and the boat. We believe with more images accurate and more training the preduction will be 100% effective on all classes. 


**Conclusion**

Must test the speed, but so far InceptionV3 is more accurate, although heavier to load.

# Getting Started

## Folders

* notebook folder: includes Jupyter notebooks with a detailed explanation on how models are trained
* model folder: includes trained models
* modules folder: includes python scripts can be stocked for experimental purposes
* test_tangram.py the main file used for inference


## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)
- Python 3.7.x, 64bit

```bash
pip install opencv-python tensorflow
```

## Get more models

The trained models are available in the `models folder:

```
cd models/
```

**Inception V3**

```
wget -O InceptionV3.h5 https://drive.google.com/uc?export=download&id=1dUDxDI5Hg-bG4B54ORTfTJyV9YIT3PHc
```

**MobileNetV2**

```
wget -O MobileNetV2.h5 https://drive.google.com/uc?export=download&id=13dDtd4jsCyA6Z4MEPK3RsWDLiCZJvEPc
```
## Inference

All model files can be found in the models folder. To use a model for inference, either connect the camera to your device or select a video file and write the following command line:

```
python test_tangram -c [camera] -s [side : left | right] -o [output_folder] -m [model] -i [input folder (OPTIONAL)]
```

**Example:**

```
python test_tangram.py -c 1 -s left -o result_pics -m models\tangram_jason_mobilenet_final_06082020.h5
```


# Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD](https://github.com/jenguehard)
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
