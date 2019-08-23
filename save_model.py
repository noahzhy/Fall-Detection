import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

tf.compat.v1.reset_default_graph()
# saver = tf.train.Saver()

# history = dict(train_loss=[],
#                      train_acc=[],
#                      test_loss=[],
#                      test_acc=[])

sess=tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

# predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})


history = pickle.load(open("history.p", "rb"))
predictions = pickle.load(open("predictions.p", "rb"))

pickle.dump(predictions, open("predictions.p", "wb"))
pickle.dump(history, open("history.p", "wb"))
tf.io.write_graph(sess.graph_def, '.', 'media/checkpoint/3_100_bs32.pbtxt')
saver.save(sess, save_path = "media/checkpoint/3_100_bs32.ckpt")
sess.close()
