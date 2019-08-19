import cv2
import minpy.numpy as np
import pandas as pd


def kalman_filter(data):
    kalman = cv2.KalmanFilter(6, 6)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 1]], np.float32) * 1

    row_num = data.acc_x.size

    for i in range(row_num):
        correct = np.array(data.iloc[i, 0:6].values, np.float32).reshape([6, 1])
        kalman.correct(correct)
        predict = kalman.predict()
        data.iloc[i, 0] = predict[0]
        data.iloc[i, 1] = predict[1]
        data.iloc[i, 2] = predict[2]
        data.iloc[i, 3] = predict[3]
        data.iloc[i, 4] = predict[4]
        data.iloc[i, 5] = predict[5]

    return data


def show_data(data, name=None):
    '''
    show data
    :param data: DataFrame
    :return:
    '''
    num = data.acc_x.size

    x = np.arange(num)
    fig = plt.figure(1, figsize=(100, 60))
    # 子表1绘制加速度传感器数据
    plt.subplot(2, 1, 1)
    plt.title('Origin')
    plt.plot(x, data.acc_x, label='x')
    plt.plot(x, data.acc_y, label='y')
    plt.plot(x, data.acc_z, label='z')

    # 添加解释图标
    plt.legend()
    x_flag = np.arange(0, num, num / 10)
    plt.xticks(x_flag)

    # 子表2绘制陀螺仪传感器数据
    plt.subplot(2, 1, 2)
    plt.title('After kalman')
    plt.plot(x, data.gyro_x, label='x')
    plt.plot(x, data.gyro_y, label='y')
    plt.plot(x, data.gyro_z, label='z')

    plt.legend()
    plt.xticks(x_flag)
    #plt.show()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()
