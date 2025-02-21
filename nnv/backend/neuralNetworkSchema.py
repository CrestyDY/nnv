import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generateSchema(testNumber, trainingImagesCount, iterations, numOfHiddenLayers, numOfData, neuronCount, activationFunction, outputActivationFunction):
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
    inputNeuronsToShow = neuronCount
    for i in range(inputNeuronsToShow):
        nodeName = getNodeName(0, i, "Input")
        nodeValue = float(sampleImage[0, i])
        G.add_node(nodeName, layer = 0, type = "Input", value = nodeValue, pos = (0, i))
    G.add_node("inputDots", layer = 0, type = "dots", pos = (0, inputNeuronsToShow + 1))

    for layerIndex, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        layerActivations = intermediatePredictions[layerIndex][0]

        nNeurons = layer.units
        neuronsToShow = min(inputNeuronsToShow, nNeurons)

        for i in range(neuronsToShow):
            nodeName = getNodeName(layerIndex + 1, i)
            nodeValue = float(layerActivations[i])
            if layerIndex == len(model.layers) - 1:
                nodeValue = finalPrediction[0][i]
            G.add_node(nodeName, layer = layerIndex + 1, type = "neuron", bias = float(biases[i]), value = nodeValue, pos = (layerIndex + 1, i))

        if neuronsToShow < nNeurons:
            G.add_node(f"L{layerIndex + 1}_dots", layer=layerIndex + 1,
                 type="dots", pos=(layerIndex + 1, neuronsToShow + 1))

        prevNeurons = inputNeuronsToShow if layerIndex == 0 else min(10, model.layers[layerIndex - 1].units)

        for i in range(prevNeurons):
            for j in range(neuronsToShow):
                if layerIndex == 0:
                    source = getNodeName(0, i, "Input")
                else:
                    source = getNodeName(layerIndex, i)
                target = getNodeName(layerIndex + 1, j)
                weight = float(weights[i,j])
                G.add_edge(source, target, weight = weight)



def getNodeName(layerIndex, neuronIndex, layerType = "neuron"):
    return f"L{layerIndex}_{layerType}_{neuronIndex}"
