import os
import time
from Dog_Cat_CNN import ecoModel

DATADIR = os.getcwd() + "/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 70

#trainingData = ecoModel.createTrainingData(DATADIR,CATEGORIES,IMG_SIZE)
#modelData = ecoModel.createModel(trainingData,IMG_SIZE)
#ecoModel.saveModel(modelData[0],modelData[1])
modelData = ecoModel.loadModel()

dense_layers = [0]
layer_sizes = [64]
conv_layers = [2]

for denseLayer in dense_layers:
    for layerSize in layer_sizes:
        for convLayer in conv_layers:
            name = "{}-conv-{}-nodes-{}-dense-{}".format(convLayer, layerSize, denseLayer, int(time.time()))
            print(name)
            ecoModel.trainCNN(name,modelData[0],modelData[1],layerSize,convLayer,denseLayer)
