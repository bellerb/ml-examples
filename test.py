import os
from Dog_Cat_CNN import ecoModel

DATADIR = os.getcwd() + "/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

trainingData = ecoModel.createTrainingData(DATADIR,CATEGORIES,IMG_SIZE)
modelData = ecoModel.createModel(trainingData,IMG_SIZE)
ecoModel.saveModel(modelData[0],modelData[1])
#print(visionModel.loadModel())
ecoModel.trainModel("Cats-vs-dogs-64x2-CNN",modelData[0],modelData[1])
