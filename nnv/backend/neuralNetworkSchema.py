import math

import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation


def getNodeName(layerIndex, neuronIndex, layerType="neuron"):
    return f"L{layerIndex}_{layerType}_{neuronIndex}"


class IntermediateDataCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_train, y_train, imageNumber):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.imageNumber = imageNumber
        self.intermediateActivations = []
        self.intermediateSoftmax = []
        self.intermediateWeights = []
        self.layer_models = []
        self.max_activation = 0.0

    def on_train_begin(self, logs=None):
        for i in range(len(self.model.layers)):
            layer_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.layers[i].output
            )
            self.layer_models.append(layer_model)

    def on_epoch_end(self, epoch, logs=None):
        activations = []
        weights_for_epoch = []
        sample_image = self.x_train[self.imageNumber:self.imageNumber + 1]

        true_label = self.y_train[self.imageNumber]
        print(f"True label for test image: {true_label}")

        softmax_output = self.model.predict(sample_image, verbose=0)
        predicted_class = np.argmax(softmax_output)
        print(f"Predicted class: {predicted_class}, confidence: {softmax_output[0][predicted_class]:.2%}")

        for i, layer_model in enumerate(self.layer_models):
            layer_output = layer_model.predict(sample_image, verbose=0)

            if i == 0 and layer_output.shape[1] > 100:
                layer_output = (layer_output - layer_output.min()) / (layer_output.max() - layer_output.min() + 1e-10)

            if i < len(self.layer_models) - 1:
                self.max_activation = max(self.max_activation, layer_output.max())

            activations.append(layer_output)

        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                weights_for_epoch.append(layer_weights[0])

        self.intermediateActivations.append(activations)
        self.intermediateSoftmax.append(softmax_output)
        self.intermediateWeights.append(weights_for_epoch)
        print(f"Epoch {epoch + 1}: Probabilities for each digit:",
              [f"{i}: {p:.2%}" for i, p in enumerate(softmax_output[0])])
        print(f"Max hidden layer activation value observed: {self.max_activation:.4f}")


def createNeuralNetwork(activationFunction, outputActivationFunction, numOfHiddenLayers, neuronCount=64,
                        learningRate=0.0005):

    initializer = tf.keras.initializers.HeNormal()

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(784,)),
        *[tf.keras.layers.Dense(10, activation=activationFunction,
                                kernel_initializer=initializer, bias_initializer=initializer)
          for _ in range(numOfHiddenLayers)],
        tf.keras.layers.Dense(10, activation=outputActivationFunction,
                              kernel_initializer=initializer, bias_initializer=initializer)
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learningRate, decay_steps=1000, decay_rate=1.2, staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def createGraph(model, x_train, imageNumber, neuronsToDisplay):
    G = nx.DiGraph()

    inputNeuronsToShow = neuronsToDisplay
    for i in range(inputNeuronsToShow):
        nodeName = getNodeName(0, i, "input")
        nodeValue = float(x_train[imageNumber, i])
        G.add_node(nodeName, layer=0, type="input", value=nodeValue, pos=(0, i))
    G.add_node("input_dots", layer=0, type="dots", pos=(0, inputNeuronsToShow + 1))

    for layerIndex, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        neurons = layer.units
        neuronsToShow = min(neuronsToDisplay, neurons)

        for i in range(neuronsToShow):
            nodeName = getNodeName(layerIndex + 1, i)
            G.add_node(nodeName, layer=layerIndex + 1, type="neuron",
                       bias=float(biases[i]), value=0, pos=(layerIndex + 1, i))

        if neuronsToShow < neurons:
            G.add_node(f"L{layerIndex + 1}_dots", layer=layerIndex + 1,
                       type="dots", pos=(layerIndex + 1, neuronsToShow + 1))

        prevNeurons = x_train.shape[1] if layerIndex == 0 else model.layers[layerIndex - 1].units
        prevNeuronsToShow = min(neuronsToDisplay, prevNeurons)

        for i in range(prevNeuronsToShow):
            for j in range(neuronsToShow):
                if layerIndex == 0:
                    source = getNodeName(0, i, "input")
                else:
                    source = getNodeName(layerIndex, i)
                target = getNodeName(layerIndex + 1, j)
                weight = float(weights[i, j])
                G.add_edge(source, target, weight=weight)

    return G



def trainModel(model, x_train, y_train, iterations, numOfData, imageNumber, checkpointDir="./checkpoints"):
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpointDir, 'model_at_epoch_{epoch:02d}.h5'),
        save_freq='epoch',
        save_weights_only=False,
        verbose=1
    )

    intermediateCallback = IntermediateDataCallback(x_train, y_train, imageNumber)

    history = model.fit(
        x_train, y_train,
        epochs=iterations,
        steps_per_epoch=numOfData,
        callbacks=[checkpoint_callback, intermediateCallback],
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    return history, intermediateCallback


def setupPlot(x_train, y_train, imageNumber, G, max_activation=None):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1], width_ratios=[1, 2, 0.1, 0.1])

    fig.patch.set_facecolor("#A9A9A9")

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x_train[imageNumber].reshape(28, 28), cmap="gray")
    ax0.set_facecolor("gray")
    ax0.set_title(f"Input MNIST Image: {y_train[imageNumber]}")
    ax0.axis("off")

    ax_hidden_cbar = fig.add_subplot(gs[0:2, 2])
    ax_hidden_cbar.set_facecolor("#A9A9A9")
    max_val = max_activation
    norm_hidden = plt.Normalize(0, max_val)
    sm_hidden = plt.cm.ScalarMappable(cmap=plt.cm.binary, norm=norm_hidden)
    cbar_hidden = plt.colorbar(sm_hidden, cax=ax_hidden_cbar)
    cbar_hidden.set_label("Hidden Layer Activation")

    ax1 = fig.add_subplot(gs[0:2, 1])
    ax1.set_facecolor("#A9A9A9")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0:2, 3])
    ax2.set_facecolor("#A9A9A9")
    norm_prob = plt.Normalize(0, 1)
    sm_prob = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_prob)
    cbar_prob = plt.colorbar(sm_prob, cax=ax2)
    cbar_prob.set_label("Output Probability")

    ax3 = fig.add_subplot(gs[1, 0:1])
    ax3.set_facecolor("#A9A9A9")
    ax3.axis("off")

    pos = nx.get_node_attributes(G, "pos")

    return fig, ax0, ax1, ax2, ax3, pos


def updateFrame(frame, ax1, ax3, G, pos, intermediateActivations, intermediateSoftmax,
                intermediateWeights, x_train, imageNumber, numOfHiddenLayers, max_activation=None):
    ax1.clear()
    ax1.set_facecolor("#A9A9A9")
    ax1.axis("off")

    ax3.clear()
    ax3.set_facecolor("#A9A9A9")
    ax3.axis("off")

    iterationText = f"Epoch: {frame + 1}/{len(intermediateSoftmax)}"
    ax3.text(0.1, 0.9, iterationText, fontsize=12, color="white")

    activations = intermediateActivations[frame]
    weights = intermediateWeights[frame]
    softmaxProbs = intermediateSoftmax[frame][0]

    nodeColors = []
    nodeLayerTypes = []
    nodeValues = []

    for node in G.nodes():
        if "dots" in node:
            nodeColors.append(0)
            nodeLayerTypes.append("hidden")
            nodeValues.append(0)
            continue

        if node.startswith("L"):
            parts = node.split("_")
            if parts[-1].isdigit():
                layerIndex = int(parts[0][1:]) - 1
                neuronIndex = int(parts[-1])

                if layerIndex == -1:
                    inputIndex = neuronIndex
                    input_value = float(x_train[imageNumber, inputIndex])
                    nodeColors.append(input_value)
                    nodeLayerTypes.append("input")
                    nodeValues.append(input_value)
                elif layerIndex == numOfHiddenLayers:
                    if neuronIndex < len(softmaxProbs):
                        activation = float(softmaxProbs[neuronIndex])
                        G.nodes[node]["value"] = activation
                        nodeColors.append(activation)
                        nodeLayerTypes.append("output")
                        nodeValues.append(activation)
                    else:
                        nodeColors.append(0)
                        nodeLayerTypes.append("output")
                        nodeValues.append(0)
                else:
                    # Hidden layers
                    if layerIndex < len(activations) and neuronIndex < activations[layerIndex].shape[1]:
                        activation = float(activations[layerIndex][0, neuronIndex])
                        G.nodes[node]["value"] = activation
                        nodeColors.append(activation)
                        nodeLayerTypes.append("hidden")
                        nodeValues.append(activation)
                    else:
                        nodeColors.append(0)
                        nodeLayerTypes.append("hidden")
                        nodeValues.append(0)
        else:
            nodeColors.append(0)
            nodeLayerTypes.append("hidden")
            nodeValues.append(0)

    hidden_nodes = [i for i, t in enumerate(nodeLayerTypes) if t == "hidden" or t == "input"]
    output_nodes = [i for i, t in enumerate(nodeLayerTypes) if t == "output"]

    nodes_list = list(G.nodes())

    actual_max = max(np.array(nodeValues)[hidden_nodes]) if hidden_nodes else 1.0
    max_val = 5*math.ceil(max_activation/5) if max_activation is not None else min(actual_max * 1.2, 5.0)

    if hidden_nodes:
        hidden_node_list = [nodes_list[i] for i in hidden_nodes]
        hidden_node_colors = [nodeValues[i] for i in hidden_nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=hidden_node_list, node_size=1000,
                               node_color=hidden_node_colors, cmap=plt.cm.binary,
                               vmin=0, vmax=max_val, alpha=0.7)

    if output_nodes:
        output_node_list = [nodes_list[i] for i in output_nodes]
        output_node_colors = [nodeValues[i] for i in output_nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=output_node_list, node_size=1000,
                               node_color=output_node_colors, cmap=plt.cm.viridis,
                               vmin=0, vmax=1.0, alpha=0.7)

    edges = nx.draw_networkx_edges(G, pos, ax=ax1,
                                   edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   edge_cmap=plt.cm.RdYlBu, width=1, alpha=0.3)

    labels = {}
    for i, node in enumerate(G.nodes()):
        if "dots" in node:
            labels[node] = "Remaining nodes"
        elif node.startswith(f"L{numOfHiddenLayers + 1}"):
            parts = node.split("_")
            if parts[-1].isdigit():
                node_idx = int(parts[-1])
                if node_idx < len(softmaxProbs):
                    prob = softmaxProbs[node_idx]
                    labels[node] = f"Digit {node_idx}\n{prob:.1%}"
        else:
            value = nodeValues[i]
            labels[node] = f"{value:.2f}"

    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)

    unique_x_positions = sorted(set(p[0] for p in pos.values()))
    y_min = min(p[1] for p in pos.values()) - 1

    for x_pos in unique_x_positions:
        layer_index = int(x_pos)
        if layer_index == 0:
            layer_name = "Input Layer"
        elif layer_index == len(unique_x_positions) - 1:
            layer_name = "Output Layer"
        else:
            layer_name = f"Hidden Layer #{layer_index}"

        ax1.text(x_pos, y_min, layer_name,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=8, fontweight='bold', color='black')

    probTextLeft = ""
    probTextRight = ""
    for digit in range(10):
        prob = softmaxProbs[digit]
        if digit < 5:
            probTextLeft += f"Digit {digit}: {prob:.1%}\n"
        else:
            probTextRight += f"Digit {digit}: {prob:.1%}\n"

    ax3.text(0.05, 0.4, probTextLeft, fontsize=8, color='white')
    ax3.text(0.55, 0.4, probTextRight, fontsize=8, color='white')

    max_act_value = max(nodeValues) if nodeValues else 0
    ax3.text(0.05, 0.1, f"Max activation: {max_act_value:.2f}", fontsize=8, color='white')

    return edges,

def createAnimation(fig, G, pos, intermediateActivations, intermediateSoftmax,
                    intermediateWeights, x_train, imageNumber, numOfHiddenLayers,
                    max_activation=None, checkpointDir="./checkpoints"):
    ax1 = fig.axes[2]
    ax3 = fig.axes[4]

    ani = FuncAnimation(
        fig,
        updateFrame,
        frames=len(intermediateSoftmax),
        interval=2000,
        blit=False,
        fargs=(ax1, ax3, G, pos, intermediateActivations, intermediateSoftmax,
               intermediateWeights, x_train, imageNumber, numOfHiddenLayers, max_activation)
    )

    plt.tight_layout()

    animation_path = os.path.join(checkpointDir, 'neural_network_animation.mp4')
    ani.save(animation_path, writer='ffmpeg', fps=1)
    print(f"Animation saved to {animation_path}")

    return ani

def generateSchema(testNumber=15, iterations=50,
                   numOfHiddenLayers=1, numOfData=250, neuronCount=64,
                   activationFunction='relu', outputActivationFunction='softmax',
                   neuronsToDisplay=10, learningRate=0.0005, showAnimation=True):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 784)) / 255.0

    model = createNeuralNetwork(
        activationFunction,
        outputActivationFunction,
        numOfHiddenLayers,
        neuronCount,
        learningRate
    )

    history, intermediateCallback = trainModel(
        model,
        x_train,
        y_train,
        iterations,
        numOfData,
        testNumber
    )

    G = createGraph(model, x_train, testNumber, neuronsToDisplay)

    max_activation = intermediateCallback.max_activation
    print(f"Maximum activation value detected: {max_activation:.4f}")

    max_activation = np.ceil(max_activation)

    fig, ax0, ax1, ax2, ax3, pos = setupPlot(x_train, y_train, testNumber, G, max_activation)

    ani = createAnimation(
        fig,
        G,
        pos,
        intermediateCallback.intermediateActivations,
        intermediateCallback.intermediateSoftmax,
        intermediateCallback.intermediateWeights,
        x_train,
        testNumber,
        numOfHiddenLayers,
        max_activation
    )

    if showAnimation:
        plt.show()

    return {
        "model": model,
        "history": history,
        "animation_path": os.path.join("./checkpoints", 'neural_network_animation.mp4'),
        "graph": G,
        "figure": fig,
        "max_activation": max_activation
    }

if __name__ == "__main__":
    result = generateSchema(
        testNumber=120,
        iterations= 25,
        numOfHiddenLayers= 2,
        numOfData=500,
        neuronCount=32,
        activationFunction='tanh',
        outputActivationFunction='softmax'
    )