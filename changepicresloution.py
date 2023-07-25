"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/11/7 15:10 
"""

import cv2
import os

if __name__ == '__main__':
    curDir = os.curdir
    sourceDir = os.path.join(
        curDir, 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\original_data\\face_data')
    resultDir = os.path.join(
        curDir, 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\original_data_2\\face_data')
    img_list = os.listdir(sourceDir)

    for i in img_list:
        pic = cv2.imread(os.path.join(sourceDir, i), cv2.IMREAD_COLOR)
        pic_n = cv2.resize(pic, (24, 24))
        pic_name = i
        cv2.imwrite(os.path.join(resultDir, i), pic_n)
