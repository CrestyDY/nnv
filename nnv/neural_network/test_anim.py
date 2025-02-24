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
initial_learning_rate = 0.0005
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
        sample_image = x_train[15:16]  # Keep the same test image

        # Get the true label for debugging
        true_label = y_train[15]
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

        intermediate_activations.append(activations)
        intermediate_softmax.append(softmax_output)  # Store raw softmax output
        intermediate_weights.append(weights_for_epoch)
        print(f"Epoch {epoch + 1}: Probabilities for each digit:",
              [f"{i}: {p:.2%}" for i, p in enumerate(softmax_output[0])])

# Add the callback to model.fit
intermediate_callback = IntermediateDataCallback()

# Increase epochs to allow more time for learning
history = model.fit(x_train, y_train,
                    epochs=50,  # Increased epochs
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

    # Update iteration counter in top-left corner
    ax3.clear()
    ax3.set_facecolor('#A9A9A9')
    ax3.axis('off')

    # Place the epoch count text above the two columns of accuracies
    iteration_text = f"Epoch: {frame + 1}/{len(intermediate_softmax)}"
    ax3.text(0.1, 0.9, iteration_text, fontsize=12, color='white')

    # Get data for the current frame
    activations = intermediate_activations[frame]
    weights = intermediate_weights[frame]
    softmax_probs = intermediate_softmax[frame][0]  # Now correctly shaped (10,)

    # Update node colors based on activation values
    node_colors = []
    for node in G.nodes():
        if 'dots' in node:
            node_colors.append(0)
            continue

        if node.startswith('L'):
            parts = node.split('_')
            if parts[-1].isdigit():
                layer_idx = int(parts[0][1:]) - 1
                if layer_idx < len(activations):
                    neuron_idx = int(parts[-1])
                    if neuron_idx < len(activations[layer_idx][0]):
                        if layer_idx == len(activations) - 1:  # Output layer
                            # Use actual softmax probabilities
                            activation = float(softmax_probs[neuron_idx])
                        else:
                            # Use normalized activations for hidden layers
                            activation = float(activations[layer_idx][0][neuron_idx])
                        G.nodes[node]['value'] = activation
                        node_colors.append(activation)
        elif 'input' in node and not 'dots' in node:
            input_idx = int(node.split('_')[-1])
            node_colors.append(float(x_train[15, input_idx]))
        else:
            node_colors.append(0)

    # Draw updated graph
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=1000,
                                   node_color=node_colors, cmap=plt.cm.viridis,
                                   vmin=0, vmax=1, alpha=0.7)

    # Update labels with correct formatting
    labels = {}
    for node in G.nodes():
        if 'dots' in node:
            labels[node] = '...'
        elif node.startswith('L3'):  # Output layer
            parts = node.split('_')
            if parts[-1].isdigit():
                node_idx = int(parts[-1])
                if node_idx < len(softmax_probs):
                    prob = softmax_probs[node_idx]
                    labels[node] = f"Digit {node_idx}\n{prob:.1%}"
        else:
            value = G.nodes[node].get('value', 0)
            labels[node] = f"{value:.2f}"

    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)

    # [Previous edge visualization code remains the same]

    # Update probability text with better formatting
    prob_text_left = ""
    prob_text_right = ""
    for digit in range(10):
        prob = softmax_probs[digit]
        if digit < 5:
            prob_text_left += f"Digit {digit}: {prob:.1%}\n"
        else:
            prob_text_right += f"Digit {digit}: {prob:.1%}\n"

    # Position the probability columns significantly lower and reduce font size
    ax3.text(0.05, 0.4, prob_text_left, fontsize=8, color='white')  # Left column
    ax3.text(0.55, 0.4, prob_text_right, fontsize=8, color='white')  # Right column

    return nodes,

# Create and save animation
ani = FuncAnimation(fig, update, frames=len(intermediate_softmax),
                    interval=2000, blit=False)
plt.tight_layout()
plt.show()

animation_path = os.path.join(checkpoint_dir, 'neural_network_animation_softmax.mp4')
ani.save(animation_path, writer='ffmpeg', fps=1)
print(f"Animation saved to {animation_path}")