import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

class ecoModel():

    def createTrainingData(datDir,categories,imgSize):
        trainingData = []

        for category in categories:  # do dogs and cats

            path = os.path.join(datDir,category)  # create path to dogs and cats
            classification = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    imgArray = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    #imgArray = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                    newArray = cv2.resize(imgArray, (imgSize, imgSize))  # resize to normalize data size
                    trainingData.append([newArray, classification])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
                #except OSError as e:
                #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                #except Exception as e:
                #    print("general exception", e, os.path.join(path,img))

        print(len(trainingData))

        random.shuffle(trainingData)
        for sample in trainingData[:10]:
            print(sample[1])

        return trainingData

    def createModel(trainingData,imgSize):
        X = []
        y = []

        for features,label in trainingData:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, imgSize, imgSize, 1)

        return [X,y]

    def saveModel(X,y):
        pickle_out = open("X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        return True

    def loadModel():
        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)

        return [X,y]

    def trainModel(name,X,y):

        X = X/255.0

        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        tensorboard = TensorBoard(log_dir="logs/{}".format(name))

        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])