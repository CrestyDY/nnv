import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

# Load and preprocess MNIST data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 784)) / 255.0

# Custom initializer with higher variance for more random initial predictions
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

# Define and compile the model with simplified architecture and custom initializer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(784,),
                         kernel_initializer=initializer,
                         bias_initializer=initializer),
    tf.keras.layers.Dense(16, activation='relu',
                         kernel_initializer=initializer,
                         bias_initializer=initializer),
    tf.keras.layers.Dense(10, activation='softmax',
                         kernel_initializer=initializer,
                         bias_initializer=initializer)
])

# Use a learning rate scheduler to gradually increase learning rate
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=1.2,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create a directory for checkpoints
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_at_epoch_{epoch:02d}.h5'),
    save_freq='epoch',
    save_weights_only=False,
    verbose=1
)

# Lists to store intermediate data
intermediate_activations = []
intermediate_softmax = []
intermediate_weights = []

# Modified callback to store intermediate data including weights
class IntermediateDataCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get intermediate activations and softmax output
        activations = []
        weights_for_epoch = []
        sample_image = x_train[15:16]
        current_model = tf.keras.Sequential()

        for layer in model.layers:
            current_model.add(layer)
            activations.append(current_model.predict(sample_image, verbose=0))
            layer_weights = layer.get_weights()
            if layer_weights:
                weights_for_epoch.append(layer_weights[0])

        softmax_output = model.predict(sample_image, verbose=0)

        intermediate_activations.append(activations)
        intermediate_softmax.append(softmax_output)
        intermediate_weights.append(weights_for_epoch)
        print(f"Epoch {epoch + 1}: Stored activations, weights, and softmax probabilities")

# Add the callback to model.fit
intermediate_callback = IntermediateDataCallback()

# Increase epochs to allow more time for learning
history = model.fit(x_train, y_train,
                    epochs=20,  # Increased epochs
                    steps_per_epoch=250,  # Reduced steps per epoch
                    callbacks=[checkpoint_callback, intermediate_callback],
                    batch_size=64,  # Increased batch size
                    validation_split=0.2,
                    verbose=1)

# Create a directed graph
G = nx.DiGraph()

# Helper function to create node names
def get_node_name(layer_idx, neuron_idx, layer_type="neuron"):
    return f"L{layer_idx}_{layer_type}_{neuron_idx}"

# Add input nodes (subset for visibility)
input_neurons_to_show = 10
for i in range(input_neurons_to_show):
    node_name = get_node_name(0, i, "input")
    node_value = float(x_train[15, i])
    G.add_node(node_name, layer=0, type="input", value=node_value, pos=(0, i))
G.add_node("input_dots", layer=0, type="dots", pos=(0, input_neurons_to_show + 1))

# Process each layer
for layer_idx, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    n_neurons = layer.units
    neurons_to_show = min(10, n_neurons)  # Only show up to 10 neurons per layer

    # Add nodes for neurons in this layer
    for i in range(neurons_to_show):
        node_name = get_node_name(layer_idx + 1, i)
        G.add_node(node_name, layer=layer_idx + 1, type="neuron",
                   bias=float(biases[i]), value=0, pos=(layer_idx + 1, i))

    # Add dots node if we're not showing all neurons
    if neurons_to_show < n_neurons:
        G.add_node(f"L{layer_idx + 1}_dots", layer=layer_idx + 1,
                   type="dots", pos=(layer_idx + 1, neurons_to_show + 1))

    # Add edges with weights
    prev_neurons = input_neurons_to_show if layer_idx == 0 else min(10, model.layers[layer_idx - 1].units)

    for i in range(prev_neurons):
        for j in range(neurons_to_show):
            if layer_idx == 0:
                source = get_node_name(0, i, "input")
            else:
                source = get_node_name(layer_idx, i)
            target = get_node_name(layer_idx + 1, j)
            weight = float(weights[i, j])
            G.add_edge(source, target, weight=weight)

# Set up the visualization with proper subplot layout
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], width_ratios=[1, 2, 0.1])

fig.patch.set_facecolor('#A9A9A9')

# Draw the original MNIST image
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(x_train[15].reshape(28, 28), cmap='gray')
ax0.set_facecolor('gray')
ax0.set_title("Input MNIST Image")
ax0.axis('off')

# Draw the neural network
ax1 = fig.add_subplot(gs[0:2, 1])
ax1.set_facecolor('#A9A9A9')
ax1.axis('off')

# Add colorbar
ax2 = fig.add_subplot(gs[0:2, 2])
ax2.set_facecolor('#A9A9A9')
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
plt.colorbar(sm, cax=ax2, label="Neuron Activation Value")

# Add subplot for iteration counter and probabilities
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#A9A9A9')
ax3.axis('off')

# Get node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw the nodes with color based on activation value
nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=1000,
                               node_color=[G.nodes[node].get('value', 0) for node in G.nodes()],
                               cmap=plt.cm.viridis, vmin=0, vmax=1, alpha=0.7)

# Draw edges with color based on weight
edges = nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                               edge_cmap=plt.cm.RdYlBu, width=1, alpha=0.3)

# Add labels with activation values
labels = nx.draw_networkx_labels(G, pos, {node: f"{G.nodes[node].get('value', 0):.2f}" for node in G.nodes()},
                                 ax=ax1, font_size=8)

def update(frame):
    ax1.clear()
    ax1.set_facecolor('#A9A9A9')
    ax1.axis('off')

    # Clear and update iteration counter
    ax3.clear()
    ax3.set_facecolor('#A9A9A9')
    ax3.axis('off')
    iteration_text = f"Epoch: {frame + 1}/10"
    ax3.text(0.1, 0.8, iteration_text, fontsize=12, color='white')

    # Get data for the current frame
    activations = intermediate_activations[frame]
    weights = intermediate_weights[frame]
    softmax_probs = intermediate_softmax[frame][0]  # Get probabilities for current frame

    # Update node colors based on activation values
    node_colors = []
    for layer_idx, layer_activations in enumerate(activations):
        neurons_to_show = min(10, len(layer_activations[0]))
        for neuron_idx in range(neurons_to_show):
            node_name = get_node_name(layer_idx + 1, neuron_idx)
            if node_name in G.nodes:
                activation = layer_activations[0][neuron_idx]
                G.nodes[node_name]['value'] = float(activation)
                node_colors.append(float(activation))

    # Normalize node colors
    node_colors = [G.nodes[node].get('value', 0) for node in G.nodes()]

    # Draw updated graph
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=1000,
                                   node_color=node_colors, cmap=plt.cm.viridis,
                                   vmin=0, vmax=1, alpha=0.7)

    # Update labels - show probabilities for output layer
    labels = {}
    for node in G.nodes():
        if node.startswith('L3'):  # Output layer
            node_idx = int(node.split('_')[-1])
            if node_idx < len(softmax_probs):
                prob = softmax_probs[node_idx]
                labels[node] = f"Digit {node_idx}\n{prob:.1%}"
        else:
            labels[node] = f"{G.nodes[node].get('value', 0):.2f}"

    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)

    # Update edge colors based on weights
    edge_colors = []
    for layer_idx, layer_weights in enumerate(weights):
        prev_neurons = min(10, layer_weights.shape[0])
        neurons_to_show = min(10, layer_weights.shape[1])
        for i in range(prev_neurons):
            for j in range(neurons_to_show):
                if layer_idx == 0:
                    source = get_node_name(0, i, "input")
                else:
                    source = get_node_name(layer_idx, i)
                target = get_node_name(layer_idx + 1, j)
                if G.has_edge(source, target):
                    weight = float(layer_weights[i, j])
                    G.edges[source, target]['weight'] = weight
                    edge_colors.append(weight)

    # Normalize edge colors
    if edge_colors:
        norm = plt.Normalize(min(edge_colors), max(edge_colors))
        edge_colors = [plt.cm.RdYlBu(norm(G.edges[u, v]['weight'])) for u, v in G.edges()]
    else:
        edge_colors = 'black'

    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors, width=1, alpha=0.3)

    # Add probability bar chart
    prob_text = "Digit Probabilities:\n"
    for digit, prob in enumerate(softmax_probs):
        prob_text += f"{digit}: {prob:.1%}  "
        if digit % 2 == 1:
            prob_text += "\n"
    ax3.text(0.1, 0.2, prob_text, fontsize=10, color='white')

    return nodes,


# Create and save animation with slower speed
ani = FuncAnimation(fig, update, frames=len(intermediate_softmax), interval=2000,
                    blit=False)  # Increased interval to 2000ms
plt.tight_layout()
plt.show()

animation_path = os.path.join(checkpoint_dir, 'neural_network_animation_softmax.mp4')
ani.save(animation_path, writer='ffmpeg', fps=1)  # Reduced fps to 1
print(f"Animation saved to {animation_path}")
