"""
    Import libraries
"""
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob2

train_images = []
train_labels = []
test_images = []
test_labels = []
val_images = []
val_labels = []
history_per_analysis = []
test_per_analysis = []
layers_limit = 5
epochs_limit = 7
running_analysis = True
FASHION_LABELS = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }


def preload__data():
    global train_images
    global train_labels
    global test_images
    global test_labels
    global val_images
    global val_labels


    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # if validation set is needed
    indexs = np.arange(test_images.shape[0])
    for shuffle in range(5) : 
        indexs = np.random.permutation(indexs)
    
    test_images = test_images[indexs]
    test_labels = test_labels[indexs]

    val_images = test_images[:8000]
    val_labels = test_labels[:8000]
    test_images = test_images[8000:]
    test_labels = test_labels[8000:]
    
    val_images = np.array(val_images, dtype=np.float32) / 255
    
    # Scaling images
    train_images = np.array(train_images, dtype=np.float32) / 255
    test_images = np.array(test_images, dtype=np.float32) / 255


def layer__vs__accuracy(feature_extraction) : 
    global history_per_analysis
    global test_per_analysis
    global train_images
    global test_images
    global val_images
    current_layers = 1
    
    while(current_layers <= layers_limit):
        model = keras.Sequential()
        
        if(feature_extraction):
            # Reshaping our images in order to use them in our CNN
            train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
            val_images = val_images.reshape((val_images.shape[0], 28, 28, 1))
            test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

            model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu',  input_shape=(28,28,1)))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(keras.layers.Flatten())
        else :
            model.add(keras.layers.Flatten(input_shape=(28, 28))) # Flatten layer for getting pixels as one column
        
        inner_counter = 1
        while(inner_counter <= current_layers):
            model.add(keras.layers.Dense(128, activation = 'relu')) # Adding fully-connected layers for classifying pixels
            inner_counter += 1

        model.add(keras.layers.Dense(10)) # Finally, adding a fully-connected layer for showing results

        model.compile(optimizer = 'adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])

        ## TODO: add validation for val_images
        results = model.fit(train_images, train_labels, batch_size=32, validation_data=(val_images, val_labels), epochs = epochs_limit)
        history_per_analysis.append(results.history)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
        test_per_analysis.append(test_acc)

        current_layers += 1
    
    show__accuracy__graph()

    
def show__accuracy__graph():
    ## Graphs for accuracy per added layer
    layers_accuracy = 1
    legends_accuracy = []
    for history in history_per_analysis: 
        plt.plot(list(range(1, epochs_limit + 1)), history['accuracy'])
        legends_accuracy.append('Dense layers - ' + str(layers_accuracy))
        layers_accuracy += 1
    
    plt.legend(legends_accuracy, loc = 'upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

    ## Graphs for accuracy's vdalite date per adder layer
    layers_val_accuracy = 1
    legends_val_accuracy = []
    for history in history_per_analysis: 
        plt.plot(list(range(1, epochs_limit + 1)), history['val_accuracy'])
        legends_val_accuracy.append('Dense layers - ' + str(layers_val_accuracy))
        layers_val_accuracy += 1
    
    plt.legend(legends_val_accuracy, loc = 'upper left')
    plt.ylabel('Validate accuracy')
    plt.xlabel('Epochs')
    plt.show()

    ## Accuracy of testing
    plt.plot(list(range(1, layers_limit + 1)), test_per_analysis)
    plt.ylabel('Accuracy')
    plt.xlabel('Layers')
    plt.show()

def load__and__predict(trained_cnn):
    current_cnn = []
    if (trained_cnn == '1'):
        current_cnn = tf.keras.models.load_model("./models/basic_cnn.h5")
    elif (trained_cnn == '2'):
        current_cnn = tf.keras.models.load_model("./models/deeper_dilated_cnn.h5")
    elif (trained_cnn == '3'):
        current_cnn = tf.keras.models.load_model("./models/deeper_class_with_dropouts.h5")
    elif (trained_cnn == '4'):
        current_cnn = tf.keras.models.load_model("./models/deeper_batchnorm.h5")
    
    current_cnn.compile(optimizer = 'adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])

    for filename, (version,) in glob2.iglob('./images/*.jpg', with_matches=True):    
        img = cv2.imread(filename)    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img, dtype=np.float32) / 255
        reshaped = np.reshape(img, (1, 28, 28, 1))
        print("Testing ", filename)
        print("Prediction: ")
        prediction = current_cnn.predict(reshaped)
        print(FASHION_LABELS[np.argmax(prediction[0])])

