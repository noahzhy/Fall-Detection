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
                                           [0, 0, 1]], np.float32) * 0.05

    row_num = data.x_axis.size

    for i in range(row_num):
        correct = np.array(data.iloc[i, 2:5].values, np.float32).reshape([3, 1])
        kalman.correct(correct)
        predict = kalman.predict()
        data.iloc[i, 2] = predict[0]
        data.iloc[i, 3] = predict[1]
        data.iloc[i, 4] = predict[2]
        # data.iloc[i, 3] = predict[3]

    return data


def show_data(data, name=None):
    # data = pd.read_csv('test.csv')
    '''
    show data
    :param data: DataFrame
    :return:
    '''
    num = data.x_axis.size

    x = np.arange(num)
    fig = plt.figure(1, figsize=(50, 30))
    # 子表1绘制加速度传感器数据
    plt.subplot(2, 1, 1)
    plt.title('Origin')
    plt.plot(x, data.x_axis, label='A')
    plt.plot(x, data.y_axis, label='B')
    plt.plot(x, data.z_axis, label='C')
    # plt.plot(x, data.D, label='D')

    # 添加解释图标
    plt.legend()
    x_flag = np.arange(0, num, num / 10)
    plt.xticks(x_flag)

    tmp_data = kalman_filter(data)

    # 子表2绘制陀螺仪传感器数据
    plt.subplot(2, 1, 2)
    plt.title('After kalman')
    plt.plot(x, tmp_data.x_axis, label='A')
    plt.plot(x, tmp_data.y_axis, label='B')
    plt.plot(x, tmp_data.z_axis, label='C')
    # plt.plot(x, (tmp_data.x_axis*0.2+tmp_data.y_axis*0.4+tmp_data.z_axis*0.4), label='w+Avg')

    plt.legend()
    plt.xticks(x_flag)
    #plt.show()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()



if __name__ == '__main__':
    data = pd.read_csv('E:/Fall-Detection/dataset/fall_29.csv')
    show_data(data)
