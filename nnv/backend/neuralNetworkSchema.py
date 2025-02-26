import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

def generateSchema(testNumber, trainingImagesCount, iterations, numOfHiddenLayers, numOfData, neuronCount, activationFunction, outputActivationFunction):

    # To load a previous model: model_epoch_2 = tf.keras.models.load_model('model_at_epoch_02.h5')
    sampleImage = x_train[testNumber:testNumber+1]
    for i in range(1, iterations+1):
        model = tf.keras.models.load_model(f'model_at_epoch_{i:02d}.h5')
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
        fig = plt.giture(figsize = (15, 10))
        gs = fig.add_gridspec(1, 3, width_ratios = [1, 2, 0.1])

class IntermediateDataCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, model, x_train, y_train, intermediateActivations, intermediateSoftmax, intermediateWeights, epoch, imageNumber, logs=None):
        # Get intermediate activations and softmax output
        activations = []
        weights_for_epoch = []
        sample_image = x_train[imageNumber:imageNumber + 1]  # Keep the same test image

        # Get the true label for debugging
        true_label = y_train[imageNumber]
        print(f"True label for test image: {true_label}")

        # Get direct model prediction first
        softmax_output = model.predict(sample_image, verbose=0)
        predicted_class = np.argmax(softmax_output)
        print(f"Predicted class: {predicted_class}, confidence: {softmax_output[0][predicted_class]:.2%}")

        # Get intermediate activations
        current_model = tf.keras.Sequential()
        for layer in model.layers:
            current_model.add(layer)
            layer_output = current_model.predict(sample_image, verbose=0)

            # Store normalized activations for hidden layers
            if layer != model.layers[-1]:  # Not softmax layer
                layer_output = (layer_output - layer_output.min()) / (layer_output.max() - layer_output.min() + 1e-10)
            activations.append(layer_output)

            # Store weights
            layer_weights = layer.get_weights()
            if layer_weights:
                weights_for_epoch.append(layer_weights[0])

        intermediateActivations.append(activations)
        intermediateSoftmax.append(softmax_output)  # Store raw softmax output
        intermediateWeights.append(weights_for_epoch)
        print(f"Epoch {epoch + 1}: Probabilities for each digit:",
              [f"{i}: {p:.2%}" for i, p in enumerate(softmax_output[0])])


def initNeuralNetwork(activationFunction, outputActivationFunction, numOfHiddenLayers, iterations, numOfData, trainingImagesCount, learningRate):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((trainingImagesCount, 784))/255.0

    initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 1.)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation = activationFunction, input_shape = (784,), kernel_initializer = initializer, bias_initializer = initializer),
        *[tf.keras.layers.Dense(64, activation = activationFunction, kernel_initializer = initializer, bias_initializer = initializer) for _ in range(numOfHiddenLayers)],
        tf.keras.layers.Dense(10, activation = outputActivationFunction, kernel_initializer = initializer, bias_initializer = initializer)
    ])

    modelLearningRate = learningRate

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(modelLearningRate, decay_steps = 1000, decay_rate = 1.2, staircase = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    checkpointDir = "./checkpoints"

    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= os.path.join(checkpointDir, 'model_at_epoch_{epoch:02d}.h5'),
        save_freq='epoch',
        save_weights_only = False,
        verbose = 1
    )

    intermediateActivations = []
    intermediateSoftmax = []
    intermediateWeights = []

    intermediateCallback = IntermediateDataCallback()

    history = model.fit(x_train, y_train, epochs = iterations, steps_per_epoch = numOfData, callbacks = [checkpoint_callback, intermediateCallback], batch_size = 32, validation_split = 0.2, verbose = 1)

def initGraph(neuronCount, imageNumber, x_train, model):

    G = nx.DiGraph()
    inputNeuronsToShow = neuronCount
    for i in range(inputNeuronsToShow):
        nodeName = getNodeName(0, i, "input")
        nodeValue = float(x_train[imageNumber, i])
        G.add_node(nodeName, layer = 0, type = "input", value = nodeValue, pos = (0,i))
    G.add_node("input_dots", payer = 0, type = "dots", pos = (0, inputNeuronsToShow + 1))
    for layerIndex, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        neurons = layer.units
        neuronsToShow = min(10, neurons)

    for i in range(neuronsToShow):
        nodeName = getNodeName(layerIndex + 1, i)
        G.add_node(nodeName, layer = layerIndex + 1, type = "neuron", bias = float(biases[i]), value = 0, pos = (layerIndex + 1, i))

    if neuronsToShow < neurons:
        G.add_node(f"L{layerIndex + 1}_dots", layer = layerIndex + 1, type = "dots", pos = (layerIndex +1, neuronsToShow + 1))

    if layerIndex == 0:
        prevNeurons = inputNeuronsToShow
    else:
        prevNeurons = min(10, model.layers[layerIndex - 1].units)

    for i in range(prevNeurons):
        for j in range(neuronsToShow):
            if layerIndex == 0:
                source = getNodeName(0, i, "input")
            else:
                source = getNodeName(layerIndex, i)
            target = getNodeName(layerIndex + 1, j)
            weight = float(weights[i, j])
            G.add_edge(source, target, weight = weight)

def initPlot(imageNumber, Graph, numOfHiddenLayers, x_train):

    fig = plt.figure(figsize = (15,10))
    gs = fig.add_gridspec(2, 3, height_ratios = [4, 1], width_ratios = [1, 2, 0.1])

    fig.patch.set_facecolor("#A9A9A9")

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.im.show(x_train[imageNumber]. reshape(28,28), cmap = "gray")
    ax0.set_facecolor("gray")
    ax0.set_title("Input MNIST Image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0:2, 1])
    ax1.set_facecolor("#A9A9A9")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0:2, 2])
    ax2.set_facecolor("#A9A9A9")
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap = plt.cm.viridis, norm = norm)
    plt.colorbar(sm, cax = ax2, label = "Output Probability")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#A9A9A9")
    ax3.axis("off")

    pos = nx.get_node_attributes(Graph, "pos")

    nodes = nx.draw_networkx_nodes(Graph, pos, ax = ax1, edge_color=[Graph[u][v]['weight'] for u, v in Graph.edges()],edge_cmap=plt.cm.RdYlBu, width=1, alpha=0.3)
    labels = nx.draw_networkx_labels(Graph, pos, {node: f"{Graph.nodes[node].get('value', 0):.2f}" for node in Graph.nodes()}, ax=ax1, font_size=8)

def update(frame, ax1, ax3, intermediateSoftmax, intermediateActivations, intermediateWeights, Graph, x_train, imageNumber, numOfHiddenLayers):
    ax1.clear()
    ax1.set_facecolor("#A9A9A9")
    ax1.axis("off")

    ax3.clear()
    ax3.set_facecolor("#A9A9A9")
    ax3.axis("off")

    iterationText = f"Epoch: {frame + 1}/ {len(intermediateSoftmax)}"
    ax3.text(0.1, 0.9, iterationText, fontsize = 12, color = "white")

    activations = intermediateActivations[frame]
    weights = intermediateWeights[frame]
    softmaxProbs = intermediateSoftmax[frame][0]

    nodeColors = []
    for node in Graph.nodes():
        if "dots" in node:
            nodeColors.append(0)
            continue

        if node.startswith("L"):
            parts = node.split("_")
            if parts[-1].isdigit():
                layerIndex = int(parts[0][1:]) - 1
                if layerIndex < len(activations):
                    neuronIndex = int(parts[-1])
                    if neuronIndex < len(activations[layerIndex][0]):
                        if layerIndex == len(activations) -1:
                            activation = float(softmaxProbs[neuronIndex])
                        else:
                            activation = float(activations[layerIndex][0][neuronIndex])
                        Graph.nodes[node]["value"] = activation
                        nodeColors.append(activation)
        elif "input" in node and not "dots" in node:
            inputIndex = int(node.split("_")[-1])
            nodeColors.append(float(x_train[imageNumber, inputIndex]))
        else:
            nodeColors.append(0)

    nodes = nx.draw_networkx_nodes(Graph,ax=ax1, node_size=1000, node_color=nodeColors, cmap=plt.cm.viridis, vmin=0, vmax=1, alpha=0.7)

    labels = {}
    for node in Graph.nodes():
        if "dots" in node:
            labels[node] = "..."
        elif int(node[1]) == numOfHiddenLayers + 1:
            parts = node.split("_")
            if parts[-1].isdigit():
                node_idx = int(parts[-1])
                if node_idx < len(softmaxProbs):
                    prob = softmaxProbs[node_idx]
                    labels[node] = f"Digit {node_idx}\n{prob:.1%}"
        else:
            value = Graph.nodes[node].get("value", 0)
            labels[node] = f"{value:.2f}"
    nx.draw_networkx_labels(Graph, pos, labels, ax = ax1, font_size = 8)
    probTextLeft = ""
    probTextRight = ""
    for digit in range(10):
        prob = softmaxProbs[digit]
        if digit < 5:
            probTextLeft += f"Digit {digit}: {prob:.1%}\n"
        else:
            probTextRight += f"Digit {digit}: {prob:.1%}\n"

    ax3.text(0.05, 0.4, probTextLeft, fontsize=8, color='white')  # Left column
    ax3.text(0.55, 0.4, probTextRight, fontsize=8, color='white')  # Right column

    return nodes,
def getNodeName(layerIndex, neuronIndex, layerType = "neuron"):
    return f"L{layerIndex}_{layerType}_{neuronIndex}"
