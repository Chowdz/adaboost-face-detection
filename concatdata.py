"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/11/10 10:32 
"""

import pandas as pd
import os
import csv
from alive_progress import alive_bar


def chunks_read(path):
    reader = pd.read_csv(path, iterator=True, encoding='latin1', engine='python')
    loop = True
    chunkSize = 10000000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    df = pd.concat(chunks, ignore_index=True)
    return df


if __name__ == "__main__":
    file_path_list = []
    file_list = []
    file_path = 'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_2'
    f_list = os.listdir(file_path)
    for i in f_list:
        feature_path = file_path + '\\' + i
        file_list.append(i)
        file_path_list.append(feature_path)
    for j in range(len(file_list)):
        file_list[j] = file_list[j][:-4]

    feature_name = ['A1', 'A1', 'A2', 'A2', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'B4', 'B4', 'C1', 'C1', 'D1', 'D1']
    feature_label = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    file_name = ['A1', 'A2', 'B1', 'B2', 'B3', 'B4', 'C1', 'D1']
    concat_name = ['A1_T.csv', 'A1_F.csv', 'A2_T.csv', 'A2_F.csv',
                   'B1_T.csv', 'B1_F.csv', 'B2_T.csv', 'B2_F.csv',
                   'B3_T.csv', 'B3_F.csv', 'B4_T.csv', 'B4_F.csv',
                   'C1_T.csv', 'C1_F.csv', 'D1_T.csv', 'D1_F.csv']

    for k in range(3, len(file_path_list)):
        print(f'开始迭代第{k + 1}个')
        file_list[k] = chunks_read(file_path_list[k])
        print(f'第{k + 1}个读取完毕')
        file_list[k].columns.values[5] = 'feature_value'
        file_list[k]['feature_name'] = feature_name[k]
        file_list[k]['label'] = [feature_label[k] for i in range(len(file_list[k]))]
        print(f'第{k + 1}个修改名称和添加标注完毕')
        with open(concat_name[k], "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["feature_key", "feature_value", "label"])
            with alive_bar(len(file_list[k]), force_tty=True) as bar:
                for n in range(len(file_list[k])):
                    feature_key = str(file_list[k].iloc[n, 6]) \
                                  + "X" + str(file_list[k].iloc[n, 1]) \
                                  + "Y" + str(file_list[k].iloc[n, 2]) \
                                  + "H" + str(file_list[k].iloc[n, 3]) \
                                  + "W" + str(file_list[k].iloc[n, 4])
                    feature_value = file_list[k].iloc[n, 5]
                    label = file_list[k].iloc[n, 7]
                    csv_writer.writerow([feature_key, feature_value, label])
                    f.flush()
                    bar()
            f.close()
        print(f'第{k + 1}个合并特征位置并输出成文件完毕')
        if (k + 1) % 2 == 0:
            feature_data_concat = pd.concat([chunks_read(concat_name[k - 1]), chunks_read(concat_name[k])],
                                            ignore_index=True)
            print(f'第{k + 1}个合并完毕')
            feature_data_concat.to_csv(
                'E:\\Study\\PY\\02_Machine_Learning\\03_AdaBoost\\formal\\deal_data_3\\'
                + str(file_name[int((k + 1) / 2) - 1]) + str('.csv'), index=False)
            print(f'第{k + 1}个输出合并结果完毕')
        else:
            pass
