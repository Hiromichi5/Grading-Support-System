import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import Counter
import sys
import random

# 支援あり
eff_lista = [4,0,0,1,1,1,0,3,0,1,0,0,1,3,10,6,12,15,6,12,12,12,27,31,55,165,138,174,99,117,76,70,40,36,54,22,34,24,19,33,6]
eff_listb = [9,4,0,0,3,0,0,0,1,1,0,1,0,0,1,3,1,0,1,6,12,13,21,22,27,36,39,79,78,144,162,196,108,97,64,67,34,21,15,9,6]
eff_listc = [7,2,2,2,2,2,2,0,2,4,0,2,2,0,4,0,2,2,0,4,16,58,94,256,294,288,160,68,27,14,14,2,0,0,7,0,0,0,0,0,0]
eff_listd = [2,0,0,1,1,1,0,2,0,1,1,1,7,11,15,11,25,22,19,15,26,26,40,44,144,282,175,162,92,89,71,32,37,33,33,15,15,7,8,7,5]
eff_liste = [2,0,0,1,1,1,0,2,0,1,0,0,1,2,9,5,11,14,5,11,11,11,25,127,51,196,176,162,162,109,71,65,37,33,50,21,32,22,18,30,5]
eff_listf = [0,2,0,0,0,4,0,3,6,4,5,3,0,5,6,9,1,19,18,38,39,48,55,68,63,130,140,139,119,77,67,59,40,39,29,20,18,11,11,6]
eff_listg = [0,0,1,1,3,4,5,9,6,8,7,10,18,24,12,39,38,53,48,51,129,139,142,149,111,93,62,78,72,49,45,20,21,10,5,2,2,3,3,0]
eff_listh = [0,4,0,2,0,5,8,12,1,1,3,0,3,18,15,25,23,31,58,80,100,129,132,112,85,82,55,48,56,40,50,33,32,15,10,8,7,1,0,0]
eff_listi = [0,0,0,4,3,1,1,1,1,1,1,1,1,1,1,1,3,10,29,35,48,81,120,152,148,178,160,120,59,38,42,40,17,0,0,0,0,0,0,0]
eff_listj = [0, 2, 0, 1, 1, 1, 4, 8, 2, 4, 2, 4, 7, 15, 11, 22, 19, 34, 41, 59, 89, 105, 109, 109, 86, 101, 85, 64, 82, 45, 54, 37, 21, 31, 14, 13, 9, 5, 4, 2]
eff_listk = [0, 0, 0, 2, 3, 2, 3, 5, 3, 4, 4, 5, 9, 12, 6, 15, 21, 21, 38, 43, 98, 120, 132, 152, 129, 136, 101, 98, 75, 52, 33, 30, 19, 5, 2, 1, 3, 1, 1, 0]
eff_listl = [3,1,1,1,0,1,0,2,1,0,2,1,0,1,3,2,4,4,7,12,15,25,25,20,34,49,85,72,155,174,169,104,95,63,66,34,23,12,9,3]
eff_listm = [2,0,3,3,2,4,1,2,2,2,2,2,2,0,2,4,0,2,2,0,4,5,12,24,30,44,76,108,154,206,254,158,110,58,57,34,24,2,2,1]
eff_listn = [2, 2, 2, 4, 3, 5, 4, 5, 4, 6, 10, 12, 7, 21, 19, 27, 25, 25, 66, 72, 77, 86, 70, 68, 69, 93, 113, 127, 149, 89, 65, 34, 31, 18, 13, 2, 2, 0,1, 0]
eff_listo = [1, 1, 0, 0, 0, 1, 1, 3, 3, 2, 2, 3, 0, 3, 5, 5, 3, 14, 12, 25, 27, 36, 40, 44, 48, 89, 112, 105, 137, 125, 118, 81, 67, 51, 47, 27, 20, 11, 10, 4]
Bad = [eff_lista,eff_listb,eff_listc,eff_listd,eff_liste,eff_listf,eff_listg,eff_listh,eff_listi,eff_listj,eff_listk,eff_listl,eff_listm,eff_listn,eff_listo]
# 支援なし
eff_lista = [0,3,4,4,7,12,12,7,10,13,15,18,24,33,49,67,82,118,159,172,157,130,111,133,147,40,39,36,39,22,40,1,7,1,0,0,0,0,0,0,0]
eff_listb = [6,3,1,4,3,1,1,3,4,1,3,3,1,3,3,6,13,22,16,30,31,70,85,79,91,106,135,162,148,67,64,78,54,37,25,13,13,6,4,3,3]
eff_listc = [9,2,2,7,2,0,2,2,0,4,2,4,4,19,16,14,21,19,43,38,86,133,206,246,241,154,116,49,22,6,6,8,4,9,0,0,0,0,0,0,0]
eff_listd = [5,2,1,1,1,1,1,1,1,1,1,1,1,2,12,39,51,65,120,168,211,207,226,253,107,61,30,25,21,8,0,0,0,0,0,0,0,0,0,0,0]
eff_liste = [2,2,2,1,0,0,4,2,7,33,35,19,60,29,58,54,86,100,131,125,117,114,162,149,107,49,15,53,39,12,25,19,16,11,1,26,12,5,35,2,0]
eff_listf = [0,0,4,5,4,3,6,7,7,5,14,10,23,28,35,38,68,107,130,146,129,125,101,111,60,68,41,44,35,18,15,8,4,5,2,3,1,0,0,0]
eff_listg = [2,3,5,2,2,9,3,12,10,7,5,18,17,24,29,24,28,43,58,91,83,117,130,125,135,128,100,88,58,28,32,23,10,8,2,7,4,5,0,0]
eff_listh = [0,1,0,0,5,0,3,3,5,7,9,25,19,42,50,48,62,80,91,118,155,148,110,75,68,58,42,41,20,25,18,12,18,4,7,2,4,4,0,0]
eff_listi = [0,2,2,2,2,0,4,0,2,2,2,6,8,11,7,16,12,10,8,16,16,32,35,100,139,142,150,162,147,91,25,28,26,26,10,10,6,7,6,6]
eff_listj = [0, 1, 3, 2, 3, 4, 4, 7, 7, 6, 9, 17, 19, 31, 38, 33, 52, 76, 93, 118, 122, 130, 103, 123, 87, 84, 41, 57, 37, 20, 21, 14, 10, 5, 3, 4, 3,0,0,0]
eff_listk = [1, 1, 4, 3, 2, 4, 4, 5, 6, 5, 4, 15, 15, 17, 24, 20, 23, 26, 33, 53, 29, 76, 81, 116, 133, 136, 123, 126, 104, 64, 37, 34, 49, 16, 4, 6, 4,0,0,0]
eff_listl = [4,3,1,2,2,2,3,1,2,1,4,4,3,2,1,1,2,8,5,13,25,17,35,39,66,58,75,87,125,124,153,136,66,67,78,45,26,23,13,9]
eff_listm = [3,1,0,2,1,2,3,2,1,3,5,4,2,1,3,8,23,29,35,29,36,34,51,79,133,128,146,153,186,226,211,154,116,49,22,6,6,8,4,0]
eff_listn = [2, 2, 1, 5, 3, 7, 5, 5, 5, 11, 9, 12, 16, 16, 25, 36, 46, 60, 59, 75, 90, 102, 134, 128, 123, 100, 122, 127, 121, 88, 63, 28, 12, 6, 5,2, 2,0,0,0]
eff_listo = [2, 1, 2, 3, 3, 2, 4, 3, 2, 3, 7, 5, 13, 15, 8, 13, 25, 47, 57, 89, 67, 61, 68, 75, 63, 65, 68, 65, 86, 76, 87, 77, 36, 34, 43, 34, 15, 14, 3, 4]
Others = [eff_lista,eff_listb,eff_listc,eff_listd,eff_liste,eff_listf,eff_listg,eff_listh,eff_listi,eff_listj,eff_listk,eff_listl,eff_listm,eff_listn,eff_listo]

def calc(name):
    timestamps = []
    soundtype = []
    drawtime = []

    file_path = os.path.join(name, 'Result')

    with open(os.path.join(file_path, 'sound_output.csv'), 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if row:
                timestamps.append(float(row[0]))
                soundtype.append(row[1])

    with open(os.path.join(file_path, name + '.csv'), 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if row:
                drawtime.append(float(row[0]))

    all_value = []
    j = 0
    for i in range(len(drawtime)):
        type = soundtype[j]
        end = timestamps[j + 1]
        if drawtime[i] <= end:
            all_value.append(type)
        else:
            j += 1
            all_value.append(soundtype[j])
    return drawtime, all_value


def count_timestamps_in_window(timestamps, value, window_size):
    counts = []
    start_index = 0
    end_index = 0

    for i in range(int(timestamps[0]), int(timestamps[-1]) - window_size + 1):
        window_end = i + window_size

        while start_index < len(timestamps) and timestamps[start_index] < i:
            start_index += 1

        while end_index < len(timestamps) and timestamps[end_index] < window_end:
            end_index += 1

        tmp = value[start_index:end_index]
        if tmp:
            mode = Counter(tmp).most_common(1)[0][0]
        else:
            mode = value[start_index]
        count = end_index - start_index
        counts.append((i, mode, count))
    return counts


# 値を削除
def remove_values(arr, value, count):
    mask = np.ones(len(arr), dtype=bool)  # すべてTrueのマスクを作成
    remove_indices = np.where(arr == value)[0][:count]  # 指定した値のインデックスを取得
    mask[remove_indices] = False  # 削除するインデックスをFalseに設定
    return arr[mask]  # マスクを使って新しい配列を返す

def create_count_list(numbers):
    # NumPy配列とリストの両方に対応
    # numbers が NumPy配列の場合は、tolist()でリストに変換
    if isinstance(numbers, np.ndarray):
        numbers = numbers.tolist()
    
    # 0から40までの数字の出現回数を格納するリスト
    L = [0] * 41
    
    # 各数字の出現回数をカウント
    for num in numbers:
        # 整数値に変換（浮動小数点数の場合に対応）
        num_int = int(num)
        if 0 <= num_int <= 40:
            L[num_int] += 1
    
    return np.array(L)

def create_list(l):
    rl = []
    for i,num in enumerate(l):
        for _ in range(int(num)):
            rl.append(i)
    return np.array(rl)


def histogram_with_boxplot(Bad, Others, name, max_count, min_val, max_val):
    plt.rcParams['font.size'] = 16  # 全体のフォントサイズを大きくする
    fig, (ax_hist, ax_box) = plt.subplots(nrows=2, ncols=1,
                                          gridspec_kw={"height_ratios": [3, 1]},
                                          figsize=(6, 6))
    bins = np.linspace(min_val, max_val, 42)

    mean_bad = np.mean(Bad) + 0.5
    mean_others = np.mean(Others) + 0.5
    Bad_2 = Bad + 0.5
    Others_2 = Others + 0.5

    ax_hist.hist([Bad_2, Others_2], bins=bins, label=['Support', 'No-Support'],
                 rwidth=1,
                 alpha=0.7, density=False, color=['red', 'blue'])

    

    ax_hist.axvline(mean_bad, color='red', linestyle='dashed', linewidth=1)
    ax_hist.axvline(mean_others, color='blue', linestyle='dashed', linewidth=1)

    xticks = np.array([0.5,5.5,10.5,15.5,20.5,25.5,30.5,35.5,40.5])
    xticklabels = np.array([0,5,10,15,20,25,30,35,40])
    ax_hist.set_xticks(xticks)
    ax_hist.set_xticklabels(xticklabels)
    
    ax_hist.set_xlim(min_val - 0.5, max_val + 0.5)
    ax_box.set_xlim(min_val - 0.5, max_val + 0.5)

    ax_hist.set_ylabel('Frequency', fontsize=18)
    ax_hist.legend(fontsize=14)
    ax_hist.grid(True, alpha=0.3)
    
    # 箱ヒゲ
    box_data = [Others_2, Bad_2]
    box = ax_box.boxplot(box_data, vert=False, patch_artist=True, tick_labels=['No-Support', 'Support'],
                         widths=0.25,
                         medianprops=dict(color="black"))

    colors = ['blue', 'red']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.axvline(mean_bad, color='red', linestyle='dashed', linewidth=1)
    ax_box.axvline(mean_others, color='blue', linestyle='dashed', linewidth=1)

    ax_box.set_xlabel('Basic Work Efficiency', fontsize=18)
    ax_box.set_xticks(xticks)
    ax_box.set_xticklabels(xticklabels)
    ax_box.set_xlim(min_val - 0.5, max_val + 0.5)
    ax_box.set_xlim(min_val - 0.5, max_val + 0.5)

    ax_box.grid(True, alpha=0.3)

    ax_hist.set_xlim(min_val, max_val)
    ax_box.set_xlim(min_val, max_val)

    # 縦軸のスケールを揃える
    ax_hist.set_ylim(0, max_count)

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':

    subject = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J', 'K', 'L', 'M', 'N','O']
    number = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J', 'K', 'L', 'M', 'N','O']
    result = [['被験者','Bad','Others']]

    n = int(input("被験者の番号を入力(0~14):"))
    name = subject[n]
    num = subject[n]


    eff_list = Bad[n]
    Bad = []
    for i in range(len(eff_list)):
        for _ in range(eff_list[i]):
            Bad.append(i)
    Bad = np.array(Bad)

    eff_list = Others[n]
    Others = []
    for i in range(len(eff_list)):
        for _ in range(eff_list[i]):
            Others.append(i)
    Others = np.array(Others)
    print(len(eff_list))

    Bad = create_count_list(Bad)*1.5
    Others = create_count_list(Others)*1.5
    Bad = create_list(Bad)
    Others = create_list(Others)
    print("Support")
    print(round(np.mean(Bad),1))
    print("Non-Support")
    print(round(np.mean(Others),1))

    # 各被験者のヒストグラムとボックスプロットを表示
    mean_b = np.mean(Bad)
    mean_o = np.mean(Others)
    print(mean_b)
    print(mean_o)

    print("Bad = ",len(Bad),int(len(Bad)/60),"分",len(Bad)%60,"秒")
    print("Others = ",len(Others),int(len(Others)/60),"分",len(Others)%60,"秒")
    time_b = len(Bad)
    time_o = len(Others)

    all_max_count = 300
    global_min_val = 0
    global_max_val = 41
    histogram_with_boxplot(Bad, Others, num, all_max_count, global_min_val, global_max_val)

    # print(pd.DataFrame(result))
