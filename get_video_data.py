import os
import RGB_difference_draw as draw


path = "F:/le2i/fallv2"
def file_name():
    count = 0
    F = []
    for root, dirs, files in os.walk(path):
        #print root
        # print dirs
        for file in files:
            #print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == '.avi':
                count += 1
                t = os.path.splitext(file)[0]
                tf = '{}/{}.avi'.format(root,t)
                print(tf)   #打印所有py格式的文件名
                F.append(tf) #将所有的文件名添加到L列表中
                # os.rename(tf, path + '/' + 'video_{0:03d}.avi'.format(count))
    return F   # 返回L列表

fileList = file_name()
for file in fileList:
    print(file)
    draw.get_data_from_video(file)
