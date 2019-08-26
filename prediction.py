import os
import numpy as np
from collections import deque
import pandas as pd
import tensorflow as tf
import datetime
import time
import cv2


INPUT_NODE = "inputs"
OUTPUT_NODES = {"y_"}
OUTPUT_NODE = "y_"
INPUT_SIZE = {1, 50, 3}
OUTPUT_SIZE = 6

pb = 'media/frozen_class3_epoch120_bs32.pb'

def test_data_load():
    df = pd.read_csv('test.csv', usecols=['x-axis', 'y-axis', 'z-axis'])
    xs = df['x-axis'].values[0:50]
    ys = df['y-axis'].values[0:50]
    zs = df['z-axis'].values[0:50]
    in_data = [xs, ys, zs]
    return np.asarray(in_data, dtype= np.float32).reshape(-1, 50, 3)

sess = tf.Session()
output_graph_def = tf.GraphDef()


with open(pb,"rb") as f:
    output_graph_def.ParseFromString(f.read())
    tf.import_graph_def(output_graph_def, name="")

node_in = sess.graph.get_tensor_by_name('input:0')
model_out = sess.graph.get_tensor_by_name('y_:0')

def prediction(data):
    # pass
    feed_dict = {node_in:list([data])}
    # feed_dict = {node_in:in_data}
    pred = sess.run(model_out, feed_dict)
    res = [pred.tolist()[0].index(max(pred.tolist()[0])), max(pred.tolist()[0])]
    # print(pred)
    return res

# x = deque(maxlen=50)
# q = test_data_load()[0]

# for i in q:
#     x.append([i[0],i[1],i[2]])
# in_data.append([xs, ys, zs])
# reshaped_segments = np.asarray(in_data, dtype= np.float32).reshape(-1, 50, 3)
