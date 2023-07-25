"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/11/7 19:08 
"""

import os
import csv
import cv2 as cv
import pandas as pd
import numpy as np
from alive_progress import alive_bar


def getFileName(file_path):
    file_list = []
    f_list = os.listdir(file_path)
    for i in f_list:
        path = file_path + '\\' + i
        file_list.append(path)
    return file_list


def integral(path):
    img = pd.DataFrame(cv.imread(path, cv.IMREAD_GRAYSCALE))
    r_list_max, r_list_min = [], []
    for i in range(len(img)):
        r_max = max(img.iloc[i, :])
        r_min = min(img.iloc[i, :])
        r_list_max.append(r_max)
        r_list_min.append(r_min)
    max_value = max(r_list_max)
    min_value = min(r_list_min)
    for m in range(len(img)):
        for n in range(len(img)):
            img.iloc[m, n] = (img.iloc[m, n] - min_value) / (max_value - min_value)
    img = np.array(img)
    integral_img = pd.DataFrame(cv.integral(img))
    integral_img.drop(0, axis=0, inplace=True)
    integral_img.drop(0, axis=1, inplace=True)
    return img, integral_img


def feature_A1(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_A1"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(1, img_Height + 1):
                    for j in range(2, img_Width + 2, 2):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_a1_white = \
                                    int_img.iloc[m, n - int(j / 2)] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m - i, n - int(j / 2)] \
                                    - int_img.iloc[m, n - j]
                                f_a1_black = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - i, n - int(j / 2)] \
                                    - int_img.iloc[m, n - int(j / 2)] \
                                    - int_img.iloc[m - i, n]
                                f_a1 = f_a1_white - f_a1_black
                                csv_writer.writerow([k, m, n, i, j, f_a1])
                                f.flush()
                bar()
        f.close()
    return


def feature_A2(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_A2"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(2, img_Height + 2, 2):
                    for j in range(1, img_Width + 1):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_a2_white = \
                                    int_img.iloc[m - int(i / 2), n] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m - int(i / 2), n - j] \
                                    - int_img.iloc[m - i, n]
                                f_a2_black = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - int(i / 2), n - j] \
                                    - int_img.iloc[m - int(i / 2), n] \
                                    - int_img.iloc[m, n - j]
                                f_a2 = f_a2_white - f_a2_black
                                csv_writer.writerow([k, m, n, i, j, f_a2])
                                f.flush()
                bar()
        f.close()
    return


def feature_B1(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_B1"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(1, img_Height + 1):
                    for j in range(3, img_Width + 3, 3):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_b1_white1 = \
                                    int_img.iloc[m, n - int(2 * j / 3)] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m, n - j] \
                                    - int_img.iloc[m - i, n - int(2 * j / 3)]
                                f_b1_white2 = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - i, n - int(j / 3)] \
                                    - int_img.iloc[m, n - int(j / 3)] \
                                    - int_img.iloc[m - i, n]
                                f_b1_black = \
                                    int_img.iloc[m, n - int(j / 3)] \
                                    + int_img.iloc[m - i, n - int(2 * j / 3)] \
                                    - int_img.iloc[m, n - int(2 * j / 3)] \
                                    - int_img.iloc[m - i, n - int(j / 3)]
                            f_b1 = f_b1_white1 + f_b1_white2 - 2 * f_b1_black
                            csv_writer.writerow([k, m, n, i, j, f_b1])
                            f.flush()
                bar()
        f.close()
    return


def feature_B2(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_B2"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(3, img_Height + 3, 3):
                    for j in range(1, img_Width + 1):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_b2_white1 = \
                                    int_img.iloc[m - int(2 * i / 3), n] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m - int(2 * i / 3), n - j] \
                                    - int_img.iloc[m - i, n]
                                f_b2_white2 = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - int(i / 3), n - j] \
                                    - int_img.iloc[m, n - j] \
                                    - int_img.iloc[m - int(i / 3), n]
                                f_b2_black = \
                                    int_img.iloc[m - int(i / 3), n] \
                                    + int_img.iloc[m - int(2 * i / 3), n - j] \
                                    - int_img.iloc[m - int(i / 3), n - j] \
                                    - int_img.iloc[m - int(2 * i / 3), n]
                            f_b2 = f_b2_white1 + f_b2_white2 - 2 * f_b2_black
                            csv_writer.writerow([k, m, n, i, j, f_b2])
                            f.flush()
                bar()
        f.close()
    return


def feature_B3(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_B3"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(1, img_Height + 1):
                    for j in range(4, img_Width + 4, 4):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_b3_white1 = \
                                    int_img.iloc[m, n - int(3 * j / 4)] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m, n - j] \
                                    - int_img.iloc[m - i, n - int(3 * j / 4)]
                                f_b3_white2 = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - i, n - int(j / 4)] \
                                    - int_img.iloc[m, n - int(j / 4)] \
                                    - int_img.iloc[m - i, n]
                                f_b3_black = int_img.iloc[m, n - int(j / 4)] \
                                    + int_img.iloc[m - i, n - int(3 * j / 4)] \
                                    - int_img.iloc[m, n - int(3 * j / 4)] \
                                    - int_img.iloc[m - i, n - int(j / 4)]
                            f_b3 = f_b3_white1 + f_b3_white2 - f_b3_black
                            csv_writer.writerow([k, m, n, i, j, f_b3])
                            f.flush()
                bar()
        f.close()
    return


def feature_B4(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_B4"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(4, img_Height + 4, 4):
                    for j in range(1, img_Width + 1):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_b4_white1 = \
                                    int_img.iloc[m - int(3 * i / 4), n] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m - int(3 * i / 4), n - j] \
                                    - int_img.iloc[m - i, n]
                                f_b4_white2 = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - int(i / 4), n - j] \
                                    - int_img.iloc[m, n - j] \
                                    - int_img.iloc[m - int(i / 4), n]
                                f_b4_black = \
                                    int_img.iloc[m - int(i / 4), n] \
                                    + int_img.iloc[m - int(3 * i / 4), n - j] \
                                    - int_img.iloc[m - int(i / 4), n - j] \
                                    - int_img.iloc[m - int(3 * i / 4), n]
                            f_b4 = f_b4_white1 + f_b4_white2 - f_b4_black
                            csv_writer.writerow([k, m, n, i, j, f_b4])
                            f.flush()
                bar()
        f.close()
    return


def feature_C1(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_C1"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(3, img_Height + 3, 3):
                    for j in range(3, img_Width + 3, 3):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_c1_white = int_img.iloc[m, n] \
                                             + int_img.iloc[m - i, n - j] \
                                             - int_img.iloc[m - i, n] \
                                             - int_img.iloc[m, n - j]
                                f_c1_black = \
                                    int_img.iloc[m - int(i / 3), n - int(j / 3)] \
                                    + int_img.iloc[m - int(2 * i / 3), n - int(2 * j / 3)] \
                                    - int_img.iloc[m - int(i / 3), n - int(2 * j / 3)] \
                                    - int_img.iloc[m - int(2 * i / 3), n - int(j / 3)]
                                f_c1 = f_c1_white - 9 * f_c1_black
                                csv_writer.writerow([k, m, n, i, j, f_c1])
                                f.flush()
                bar()
        f.close()
    return


def feature_D1(file_path, output_name):
    pic_list = getFileName(file_path)
    with open(output_name, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["number", "X", "Y", "Height", "Width", "feature_D1"])
        with alive_bar(len(pic_list), force_tty=True) as bar:
            for k in range(len(pic_list)):
                print(f"deal file{k}")
                img, int_img = integral(pic_list[k])
                img_Width = img.shape[1]
                img_Height = img.shape[0]
                for i in range(2, img_Height + 2, 2):
                    for j in range(2, img_Width + 2, 2):
                        for m in range(i, img_Height):
                            for n in range(j, img_Width):
                                f_d1_white1 = \
                                    int_img.iloc[m, n] \
                                    + int_img.iloc[m - int(i / 2), n - int(j / 2)] \
                                    - int_img.iloc[m - int(i / 2), n] \
                                    - int_img.iloc[m, n - int(j / 2)]
                                f_d1_white2 = \
                                    int_img.iloc[m - int(i / 2), n - int(j / 2)] \
                                    + int_img.iloc[m - i, n - j] \
                                    - int_img.iloc[m - int(i / 2), n - j] \
                                    - int_img.iloc[m - i, n - int(j / 2)]
                                f_d1_black1 = \
                                    int_img.iloc[m - int(i / 2), n] \
                                    + int_img.iloc[m - i, n - int(j / 2)] \
                                    - int_img.iloc[m - int(i / 2), n - int(j / 2)] \
                                    - int_img.iloc[m - i, n]
                                f_d1_black2 = \
                                    int_img.iloc[m, n - int(j / 2)] \
                                    + int_img.iloc[m - int(i / 2), n - j] \
                                    - int_img.iloc[m, n - j] \
                                    - int_img.iloc[m - int(i / 2), n - int(j / 2)]
                                f_d1 = f_d1_white1 + f_d1_white2 - f_d1_black1 - f_d1_black2
                                csv_writer.writerow([k, m, n, i, j, f_d1])
                                f.flush()
                bar()
        f.close()
    return


if __name__ == "__main__":
    pic_path_1 = 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\original_data_2\\face_data'
    pic_path_2 = 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\original_data_2\\non_face_data'
    feature_A1(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_A1_face.csv')
    feature_A1(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_A1_non_face.csv')
    feature_A2(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_A2_face.csv')
    feature_A2(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_A2_non_face.csv')
    feature_B1(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B1_face.csv')
    feature_B1(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B1_non_face.csv')
    feature_B2(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B2_face.csv')
    feature_B2(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B2_non_face.csv')
    feature_B3(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B3_face.csv')
    feature_B3(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B3_non_face.csv')
    feature_B4(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B4_face.csv')
    feature_B4(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_B4_non_face.csv')
    feature_C1(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_C1_face.csv')
    feature_C1(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_C1_non_face.csv')
    feature_D1(pic_path_1,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_D1_face.csv')
    feature_D1(pic_path_2,
               'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2\\feature_D1_non_face.csv')
