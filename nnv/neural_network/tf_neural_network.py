import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess MNIST data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 784)) / 255.0

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Softmax output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Make final prediction AFTER training
sample_image = x_train[0:1]
final_prediction = model.predict(sample_image, verbose=0)

# *** KEY CHANGE: Print Softmax Probabilities ***
print("Softmax Output (Final):", final_prediction[0])
predicted_label = np.argmax(final_prediction[0])
print("Predicted Label (Final):", predicted_label)
print("Actual Label:", y_train[0])

# Verify probabilities sum to ~1 (due to floating-point limitations)
print("Sum of Probabilities:", np.sum(final_prediction[0]))

# Create intermediate layer models and get predictions
intermediate_predictions = []
current_model = tf.keras.Sequential()

for i, layer in enumerate(model.layers):
    current_model.add(layer)
    intermediate_prediction = current_model.predict(sample_image, verbose=0)
    intermediate_predictions.append(intermediate_prediction)

# Create a directed graph
G = nx.DiGraph()

# Helper function to create node names
def get_node_name(layer_idx, neuron_idx, layer_type="neuron"):
    return f"L{layer_idx}_{layer_type}_{neuron_idx}"

# Add input nodes (subset for visibility)
input_neurons_to_show = 10
for i in range(input_neurons_to_show):
    node_name = get_node_name(0, i, "input")
    node_value = float(sample_image[0, i])
    G.add_node(node_name, layer=0, type="input", value=node_value, pos=(0, i))
G.add_node("input_dots", layer=0, type="dots", pos=(0, input_neurons_to_show + 1))

# Process each layer
for layer_idx, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    layer_activations = intermediate_predictions[layer_idx][0]

    n_neurons = layer.units
    neurons_to_show = min(10, n_neurons)

    # Add nodes for neurons in this layer
    for i in range(neurons_to_show):
        node_name = get_node_name(layer_idx + 1, i)
        node_value = float(layer_activations[i]) # Activations for hidden layers
        if layer_idx == len(model.layers) - 1: # Output layer
          node_value = final_prediction[0][i]  # Probabilities for output layer
        G.add_node(node_name, layer=layer_idx + 1, type="neuron",
                 bias=float(biases[i]), value=node_value, pos=(layer_idx + 1, i))

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
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 0.1])

# Draw the original MNIST image
ax0 = fig.add_subplot(gs[0])
ax0.imshow(x_train[0].reshape(28, 28), cmap='gray')
ax0.set_title(f"Input MNIST Image\nPredicted: {predicted_label}")
ax0.axis('off')

# Draw the neural network
ax1 = fig.add_subplot(gs[1])
ax1.axis('off')

# Get node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw the nodes with color based on activation value
for node in G.nodes():
    node_type = G.nodes[node].get('type', 'neuron')
    layer = G.nodes[node].get('layer')
    if node_type == 'dots':
        ax1.plot(pos[node][0], pos[node][1], 'k.', markersize=20)
    else:
        value = G.nodes[node].get('value', 0)

        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[node], node_size=1000,
                             node_color=[value], cmap=plt.cm.viridis,
                             vmin=0, vmax=1, alpha=0.7)

# Draw edges with color based on weight
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
norm_weights = [(w - min(weights)) / (max(weights) - min(weights)) for w in weights]

for (u, v), weight in zip(edges, norm_weights):
    color = plt.cm.RdYlBu(weight)
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=[(u, v)], edge_color=[color],
                            width=1, alpha=0.3)

# Add labels with activation values
labels = {}
for node in G.nodes():
    if G.nodes[node].get('type') != 'dots':
        value = G.nodes[node].get('value', 0)
        if G.nodes[node].get('type') == 'input':
            labels[node] = f"Input\n{value:.2f}"
        else:
            labels[node] = f"Neuron\n{value:.2f}"

nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
ax1.set_title("Neural Network with Activations\nafter Final Training Iteration", pad=20, size=14)

# Add colorbar
ax2 = fig.add_subplot(gs[2])
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
plt.colorbar(sm, cax=ax2, label="Neuron Activation Value")

plt.tight_layout()
plt.show()