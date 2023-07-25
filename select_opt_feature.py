"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/11/25 16:29 
"""

from intergralimage import getFileName
from concatdata import chunks_read
import pandas as pd
import csv


file_path = 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_3'
file_list = getFileName(file_path)
print('获取文件路径成功')
opt_feature_list = ['A1_OPT_extra.csv', 'A2_OPT.csv', 'B1_OPT.csv', 'B2_OPT.csv',
                    'B3_OPT.csv', 'B4_OPT.csv', 'C1_OPT.csv', 'D1_OPT.csv']
feature_name_list = ['A1_feature_name.csv', 'A2_feature_name.csv', 'B1_feature_name.csv', 'B2_feature_name.csv',
                     'B3_feature_name.csv', 'B4_feature_name.csv', 'C1_feature_name.csv', 'D1_feature_name.csv']
for k in range(len(file_list)):
    print(f'开始读取第{k}个文件')
    file = chunks_read(file_list[k])
    print(f'第{k}个文件读取成功')
    feature_list = sorted(list(set([file['feature_key'][i] for i in file.index])))
    pd.DataFrame(feature_list).to_csv(feature_name_list[k])
    print(f'特征长度为{len(feature_list)}')
    print(f'将第{k}个文件中特征生成一个不重复列表')
    with open(opt_feature_list[k], "w", newline="", encoding='latin1') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["feature_key", "opt_feature_value", "error"])
        print('开始写入文件')
        for i in range(267, len(feature_list)):
            feature = pd.DataFrame(file.loc[file['feature_key'] == feature_list[i], ['feature_value', 'label']])
            feature.sort_values(by='feature_value', inplace=True, ascending=True)
            feature.reset_index(drop=True, inplace=True)
            sum_error_list = []
            for j in range(len(feature)):
                feature['predict_label'] = [1 if i <= j else 0 for i in range(len(feature))]
                feature['error'] = [abs(feature.iloc[i, 1] - feature.iloc[i, 2]) for i in range(len(feature))]
                sum_error = sum(feature['error']) / len(feature)
                sum_error_list.append(sum_error)
                print(feature_list[i])
                print(feature)
            min_error = min(sum_error_list)
            print('计算出最小误差')
            min_error_index = sum_error_list.index(min_error)
            print('获取最小误差的索引')
            opt_split = feature.iloc[min_error_index, 0]
            print('定位最优切分点')
            sum_error_list.clear()
            print('清空列表')
            csv_writer.writerow([feature_list[i], opt_split, min_error])
            print('写入一行成功')
            f.flush()
        f.close()
