import cv2
import numpy as np
# import minpy.numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kalman_filter(data):
    kalman = cv2.KalmanFilter(3, 3)
    kalman.measurementMatrix = np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]], np.float32) * 0.001
    kalman.measurementNoiseCov = np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]], np.float32) * 0.008

    row_num = data.A.size

    for i in range(row_num):
        correct = np.array(data.iloc[i, 0:3].values, np.float32).reshape([3, 1])
        kalman.correct(correct)
        predict = kalman.predict()
        data.iloc[i, 0] = predict[0]
        data.iloc[i, 1] = predict[1]
        data.iloc[i, 2] = predict[2]
        # data.iloc[i, 3] = predict[3]

    return data


def show_data(data, name=None):
    # data = pd.read_csv('test.csv')
    '''
    show data
    :param data: DataFrame
    :return:
    '''
    num = data.A.size

    x = np.arange(num)
    fig = plt.figure(1, figsize=(50, 30))
    # 子表1绘制加速度传感器数据
    plt.subplot(2, 1, 1)
    plt.title('Origin')
    plt.plot(x, data.A, label='A')
    plt.plot(x, data.B, label='B')
    plt.plot(x, data.C, label='C')
    # plt.plot(x, data.D, label='D')

    # 添加解释图标
    plt.legend()
    x_flag = np.arange(0, num, num / 10)
    plt.xticks(x_flag)

    tmp_data = kalman_filter(data)

    # 子表2绘制陀螺仪传感器数据
    plt.subplot(2, 1, 2)
    plt.title('After kalman')
    plt.plot(x, tmp_data.A, label='A')
    plt.plot(x, tmp_data.B, label='B')
    plt.plot(x, tmp_data.C, label='C')
    plt.plot(x, (tmp_data.A*0.2+tmp_data.B*0.5+tmp_data.C*0.3), label='w+Avg')

    plt.legend()
    plt.xticks(x_flag)
    #plt.show()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()




data = pd.read_csv('test.csv')
# #
# show_data(data)
data = kalman_filter(data)
# # data.to_csv('./dataset/train/BSC_1_1_annotated.csv', index=False)
show_data(data)
