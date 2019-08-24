import os
import pandas as pd
import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from tensorflow.python.platform import gfile
import tensorflow as tf
# from LSTM_Model import  LSTM_Model

INPUT_NODE = "inputs"
OUTPUT_NODES = {"y_"}
OUTPUT_NODE = "y_"
INPUT_SIZE = {1, 50, 3}
OUTPUT_SIZE = 6

pb = 'media/frozen_class3_epoch120_bs32.pb'

sess = tf.Session()
output_graph_def = tf.GraphDef()

with open(pb,"rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

node_in = sess.graph.get_tensor_by_name('input:0')
model_out = sess.graph.get_tensor_by_name('y_:0')

df = pd.read_csv('test.csv', usecols=['x-axis', 'y-axis', 'z-axis'])


xs = df['x-axis'].values[0:50]
ys = df['y-axis'].values[0:50]
zs = df['z-axis'].values[0:50]
in_data = []
in_data.append([xs, ys, zs])
reshaped_segments = np.asarray(in_data, dtype= np.float32).reshape(-1, 50, 3)
feed_dict = {node_in:reshaped_segments}

# feed_dict = {node_in:in_data}
pred = sess.run(model_out, feed_dict)

print(pred)
