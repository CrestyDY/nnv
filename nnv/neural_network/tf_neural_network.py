import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

# Define the model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),                      # Hidden layer
    tf.keras.layers.Dense(10, activation='softmax')                    # Output layer
])

# Create a directed graph using NetworkX
G = nx.DiGraph()

# Calculate input shapes and output shapes for each layer
input_shape = (None, 784)  # Starting input shape
shapes = []
for layer in model.layers:
    output_shape = layer.compute_output_shape(input_shape)
    shapes.append(output_shape)
    input_shape = output_shape

# Add nodes (layers) to the graph
for i, layer in enumerate(model.layers):
    layer_name = f"{layer.name} ({layer.__class__.__name__})"
    output_shape = str(shapes[i])
    G.add_node(layer_name, layer_type=layer.__class__.__name__, output_shape=output_shape)

# Add edges (connections between layers) to the graph
for i in range(len(model.layers) - 1):
    current_layer = model.layers[i]
    next_layer = model.layers[i + 1]
    current_layer_name = f"{current_layer.name} ({current_layer.__class__.__name__})"
    next_layer_name = f"{next_layer.name} ({next_layer.__class__.__name__})"
    G.add_edge(current_layer_name, next_layer_name)

# Visualize the graph
plt.figure(figsize=(12, 8))  # Increased figure size for better readability

# Create layout with more vertical separation
pos = nx.spring_layout(G, k=1, iterations=50)  # Adjusted spring layout parameters

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.7)

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

# Create labels with output shape information
labels = {node: f"{node}\nOutput: {G.nodes[node]['output_shape']}" for node in G.nodes}

# Draw labels with adjusted position and font size
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# Add title and adjust layout
plt.title("Neural Network Architecture Graph", pad=20, size=14)
plt.axis('off')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()