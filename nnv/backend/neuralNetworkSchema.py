import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generateSchema(testNumber):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()