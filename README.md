# cnn-classifier

This repository contains my first attempt to build a convolutional neural network using Python, Tensorflow and keras. I'm neither an expert in python or machine learning. 
But I want give here my contribution to the community and for the ones who are trying to learn machine learning as I am.

Please feel free to contribute to this repository, or to comment anything about coding style, enhancements on algorithm, etc.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

#### Environment
- Python 3.7
- Conda

#### Libraries
- TensorFlow
- Keras
- numpy
- cv2
- glob2

### Installing

Just git clone this repository in your local machine as

```bash
$ git clone https://github.com/cancinos/cnn-classifier.git
```

## Dataset and dataprep

For this classificators I use the well known dataset [MNIST FASHION](https://www.tensorflow.org/datasets/catalog/fashion_mnist). 
And its preparation for training was very smoothly, what I did what just to take the test images and shake them a little with a random function in order to extract
8,000 sample for validation purpose (I decided that I need a little of cross-validation on my training). 
So resuming, I take the dataset and extract three main samples: ```train, test and validation.``` After all, I normalize my pixels dividing all by 255. 
And with that I was ready to go!

Here is important to mention that if you want to add your own images to predict, you can do it! Just make that the image your adding to the ```./images/``` folder ends up
in ```.jpg``` and also that its size is 28x28.

## Deployment

I have three main functionalities on my code. The first two options are about checking how different types of CNN get trained, and performs
with a validation set of, the well known, MNIST FASHION dataset. We will get you through a one execution but I'll encourage you to test and see how it
performs.

0. **Execute it** 

  ```
  $ python convolutional_neural_net.py
  $ Welcome to your CNN Classifier! Which option do you want to perform?
  $ 1. Check a 'Layer vs. Accuracy' Analysis
  $ 2. Check a 'Layer vs. Accuracy' using CNNs and dropout layers
  $ 3. Check how on of our trained CNNs performs
  ```
    
1. **Check a 'Layer vs. Accuracy' Analysis**

Once you selected this option you will see how a neural network starts to train itself. The catch here, or what is interesting is that you can see how it runs varying the number
of hidden layers that we are adding at the classification segment of the algorithm. Here what we are doing is just presenting a dataset of matrix of pixels (all the images 
in MNIST FASHION) and ask the neural net to recognize 'where' they are without giving it much information about 'what' they are or 'what' they represent of the image. 

**Accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_accuracy_1.PNG">
</p>

As you can see in the image the model performs better when you're classifying with just two layers, and they started to converge when you use more than 15 epochs. 
That's the reason why I decide to used between one or two layers in the classification segment and just to make sure that it converges we use 25 epochs.

**Validation accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_val_accuracy_1.PNG">
</p>

Here is important to denote the poor performance of validation using 5 fully connected layers, the other models were behaving similarly.

**Test accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_test_accuracy_1.PNG">
</p>

And finally, when we test the 3-fully-connected-layer and the 5-fully-connected-layer behaved similarly, but of all them with an accuracy above the 0.85.

2. **Check a 'Layer vs. Accuracy' using CNNs and dropout layers**

This analysis was very similar to the one before, the only difference is that I added a feature extraction part, using a dropout layer and also a convolutional layer.
The main idea here is understanding how telling the model that there are important features or characteristics on every image, and that they are not just pixel in a matrix.
The results are the following:

**Accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_accuracy_2.PNG">
</p>

**Validation accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_val_accuracy_2.PNG">
</p>

**Test accuracy vs. Epochs**
<p align="center">
  <img width="623" height="473" src="https://github.com/cancinos/cnn-classifier/blob/master/layers_test_accuracy_2.PNG">
</p>

Whats interesting here is that, in contrast with the previous analysis, when I test my model I got better performance using 1, 2 or 4 layers, but a worst one with 3 and 5 layers.

3. **Check how on of our trained CNNs performs**

I, also, trained 4 models of my own in order to you to test them. Everyone follow the same data preparation but their structure were different from each other. In changed them 
bases on my criteria and on what I believed would be the best for them to predict correctly. As soon as you pick the option it will be show you 4 models that I'll describe here:

  - **Basic CNN**
    - **Code**
    ``` python
      model = keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu',  input_shape=(28,28,1)))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(256, activation = 'relu'))
      model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
      model.compile(optimizer = 'adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                            metrics = ['accuracy'])
      results = model.fit(train_images, train_labels, batch_size=32, validation_data=(val_images, val_labels), epochs = 25)
      test_loss, test_acc = model.evaluate(test_images, test_labels)
    ```
    - **Structure**
    
    Here I have two main points to be observed. First one, I added ```Conv2D ``` layers from 32 to 128 in order for the model to learn fewer filters at the beginning
    (extracting fewer details) and more of them at the end of the feature extracture segment of the model.
    And finally, the ```Adam``` optimizer that I use here, and in all the model I trained. The decision about using Adam was bases on the idea that it is a extended version of
    sgd, one of the must important difference of it is that Adam adapts the learning rate over-time and that it uses one learning rate per parameter use and not only one global
    like sgd.
    - **Results from testing**
    
    1s 452us/sample - loss: 1.6576 - accuracy: 0.8595
    
 - **Deeper dilated CNN**
    - **Code**
    ``` python
      model = keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu',  input_shape=(28,28,1)))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(512, activation = 'relu'))
      model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
      model.compile(optimizer = 'adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                            metrics = ['accuracy'])
      results = model.fit(train_images, train_labels, batch_size=32, validation_data=(val_images, val_labels), epochs = 25)
      test_loss, test_acc = model.evaluate(test_images, test_labels)
    ```
    - **Structure**
    
    Here I have only one main point to be observed. And it's basically the difference between my first model and this one. I added a ```dilation_rate```, 
    this rate will expand our kernel and extract features in a fewer detailed way. What I want with this was to really focus on the silhouette of the object.
    This dilation is highly recommended when you are learning from images that have the main object centered and its size is the same as the image, such as the 
    MNIST FASHION dataset is.
    - **Results from testing**
    
    2s 806us/sample - loss: 1.6861 - accuracy: 0.8000
    
  - **Deeper CNN classificator with dropouts**
    - **Code**
    ``` python
      model = keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu',  input_shape=(28,28,1)))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, dilation_rate=(2, 2), padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(256, activation = 'relu'))
      model.add(tf.keras.layers.Dropout(0.5))
      model.add(keras.layers.Dense(256, activation = 'relu'))
      model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
      model.compile(optimizer = 'adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                            metrics = ['accuracy'])
      results = model.fit(train_images, train_labels, batch_size=32, validation_data=(val_images, val_labels), epochs = 25)
      test_loss, test_acc = model.evaluate(test_images, test_labels)
    ```
    - **Structure**
    
    Here I tried to develop I deeper classifier and making it more decision-maker using a dropout layer with a high dropout-rate of 0.5, what I meant with this is that 
    it will have a deeper hidden-layers but it will have to be more decisive about what object the model is classifying. It is important to mention that I using ```softmax
    activation``` in my final layer in order to get more easily what was the decision. 
    - **Results from testing**
    
    3s 2ms/sample - loss: 1.9637 - accuracy: 0.4950
    
  - **Deeper CNN Classificator with batchnorm**
    - **Code**
    ``` python
      model = keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu',  input_shape=(28,28,1)))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(256, activation = 'relu'))
      model.add(tf.keras.layers.Dropout(0.5))
      model.add(keras.layers.Dense(256, activation = 'relu'))
      model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
      model.compile(optimizer = 'adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                            metrics = ['accuracy'])
      results = model.fit(train_images, train_labels, batch_size=32, validation_data=(val_images, val_labels), epochs = 25)
      test_loss, test_acc = model.evaluate(test_images, test_labels)
    ```
    - **Structure**
    
    **This one is our favorite or the one in which I have more faith.** The main key difference that we have here is that I added a ```BatchNormalization``` layer in order to 
    normalize my weights in my hidden layers, with this I gained more accurate results and also more efficient training. Another important thing is that I removed the 
    ```dilation_rate``` based on the accuracy that I got in the last two models I trained.
    - **Results from testing**
    
    1s 649us/sample - loss: 1.6532 - accuracy: 0.8035
    
    
## Authors

* **Pablo Cancinos** - *Future Data Scientist* - [cancinos](https://github.com/cancinos)
