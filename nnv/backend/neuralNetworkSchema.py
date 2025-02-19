import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generateSchema(testNumber, trainingImagesCount, iterations, numOfHiddenLayers, numOfData, activationFunction, outputActivationFunction):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((trainingImagesCount, 784))/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation = activationFunction, input_shape = (784,)),
        *[tf.keras.layers.Dense(64, activation = activationFunction) for _ in range(numOfHiddenLayers)],
        tf.keras.layers.Dense(10, activation = outputActivationFunction)
    ])

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model_at_epoch_{epoch:02d}.h5',
        save_freq='epoch'
    )

    model.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

    history = model.fit(x_train, y_train, epochs = iterations, steps_per_epoch = numOfData, callbacks = [checkpoint_callback], batch_size = 32, validation_split = 0.2, verbose = 1)

    # To load a previous model: model_epoch_2 = tf.keras.models.load_model('model_at_epoch_02.h5')

    sampleImage = x_train[testNumber:testNumber+1]
    finalPrediction = model.predict(sampleImage, verbose = 0)

    intermediatePredictions = []
    currentModel = tf.keras.Sequential()

    for i, layer in enumerate(model.layers):
        currentModel.add(layer)
        intermediatePrediction = currentModel.predict(sampleImage, verbose = 0)
        intermediatePredictions.append(intermediatePrediction)

    G = nx.DiGraph()

def getNodeName(layerIndex, neuronIndex, layerType = "neuron"):
    return f"L{layerIndex}_{layerType}_{neuronIndex}"
