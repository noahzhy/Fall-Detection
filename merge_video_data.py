import os
import pandas as pd
import csv


path = 'dataset/'
count = 0


def change_label(lists):
    new_labels = []
    for label in lists:
        if (label != 'fall') and (label != 'lying'):
            new_labels.append('notfall')
        else:
            new_labels.append(label)
    return new_labels

for i in os.listdir(path):
    count += 1
    csv_data = pd.read_csv(path + i, usecols=['action', 'timestamp', 'x-axis', 'y-axis', 'z-axis'])
    csv_data['user'] = count
    csv_data['action'] = change_label(csv_data['action'])

    csv_data['x-axis'] = round(csv_data['x-axis'], 2)
    csv_data['y-axis'] = round(csv_data['y-axis'], 2)
    csv_data['z-axis'] = round(csv_data['z-axis'], 2)
    csv_data.to_csv('data/demo.csv', mode='a', index_label=None, header=False, index=False)
    # with open('result.csv','ab') as f:
    #     f.write(fr)


print(u'合并完毕！')
