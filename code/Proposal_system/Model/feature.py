import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 平均
mean_x, mean_y, mean_z, mean_xyz = [], [], [], []

# 分散
var_x, var_y, var_z, var_xyz = [], [], [], []

# 標準偏差
std_x, std_y, std_z, std_xyz = [], [], [], []

# 最大
max_x, max_y, max_z, max_xyz = [], [], [], []

# 最小値
min_x, min_y, min_z, min_xyz = [], [], [], []

# 中央値
median_x, median_y, median_z, median_xyz = [], [], [], []

# 範囲
range_x, range_y, range_z, range_xyz = [], [], [], []

# 周波数成分
peak_frequency_x, peak_frequency_y, peak_frequency_z, peak_frequency_xyz = [], [], [], []

# パワースペクトル密度
power_spectrum_x, power_spectrum_y, power_spectrum_z, power_spectrum_xyz = [], [], [], []

# 周波数帯域のエネルギー
peak_amplitude_x, peak_amplitude_y, peak_amplitude_z, peak_amplitude_xyz = [], [], [], []

# 自己相関関数
auto_corr_x, auto_corr_y, auto_corr_z, auto_corr_xyz = [], [], [], []

# エネルギーベースの特徴量
integral_energy_x, integral_energy_y, integral_energy_z, integral_energy_xyz = [], [], [], []

label = []
move = []
list = ['マル1','マル2','マル3','マル4','マル5','マル6','マル7','マル8','マル9','マル10',
        'バツ1','バツ2','バツ3','バツ4','バツ5','バツ6','バツ7','バツ8','バツ9','バツ10']

interval_maru, interval_batu = [], []

def feature_create(df_thre):
    # 特徴量
    global label, move, interval
    for mark in list:
        file = os.path.join('acc_train','acc_data_training_'+mark+'.csv')
        df_tmp = pd.read_csv(file)
        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        start = df_thre[df_thre['マーク'] == mark]['start'].values[0]
        start = pd.to_datetime(start)
        end = df_thre[df_thre['マーク'] == mark]['end'].values[0]
        end = pd.to_datetime(end)

        x = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)]['X']
        y = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)]['Y']
        z = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)]['Z']
        xyz = np.sqrt(x**2+y**2+z**2)
        
        move.append(True)
        if 'マル' in mark:
            label.append('マル')
            interval_maru.append(len(x))
        elif 'バツ' in mark:
            label.append('バツ')
            interval_batu.append(len(x))
        else:
            label.append(np.nan)
        # print("x = ",len(x))
        df = feature(x, y, z, xyz)
    df = pd.DataFrame()
    df.index = list
    # df['mean_x'], df['mean_y'], df['mean_z'], df['mean_xyz'] = mean_x, mean_y, mean_z, mean_xyz
    # df['var_x'], df['var_y'], df['var_z'], df['var_xyz'] = var_x, var_y, var_z, var_xyz
    df['std_x'], df['std_y'], df['std_z'], df['std_xyz'] = std_x, std_y, std_z, std_xyz
    df['max_x'], df['max_y'], df['max_z'], df['max_xyz'] = max_x, max_y, max_z, max_xyz
    df['min_x'], df['min_y'], df['min_z'], df['min_xyz'] = min_x, min_y, min_z, min_xyz
    # df['median_x'], df['median_y'], df['median_z'], df['median_xyz'] = median_x, median_y, median_z, median_xyz
    df['range_x'], df['range_y'], df['range_z'], df['range_xyz'] = range_x, range_y, range_z, range_xyz
    df['peak_frequency_x'], df['peak_frequency_y'], df['peak_frequency_z'], df['peak_frequency_xyz'] = peak_frequency_x, peak_frequency_y, peak_frequency_z, peak_frequency_xyz
    df['peak_amplitude_x'], df['peak_amplitude_y'], df['peak_amplitude_z'], df['peak_amplitude_xyz'] = peak_amplitude_x, peak_amplitude_y, peak_amplitude_z, peak_amplitude_xyz
    # df['auto_corr_x'], df['auto_corr_y'], df['auto_corr_z'], df['auto_corr_xyz'] = auto_corr_x, auto_corr_y, auto_corr_z, auto_corr_xyz
    # df['integral_energy_x'], df['integral_energy_y'], df['integral_energy_z'], df['integral_energy_xyz'] = integral_energy_x, integral_energy_y, integral_energy_z, integral_energy_xyz
    df['move'] = move
    df['label'] = label
    # df.to_csv('df.csv')
    return df, np.mean(interval_maru), np.mean(interval_batu) 

def feature(x, y, z, xyz):
    # # 平均
    # mean_x.append(np.mean(x))
    # mean_y.append(np.mean(y))
    # mean_z.append(np.mean(z))
    # mean_xyz.append(np.mean(xyz))

    # # 分散
    # var_x.append(np.var(x))
    # var_y.append(np.var(y))
    # var_z.append(np.var(z))
    # var_xyz.append(np.var(xyz))

    # 標準偏差
    std_x.append(np.std(x))
    std_y.append(np.std(y))
    std_z.append(np.std(z))
    std_xyz.append(np.std(xyz))

    # 最大
    max_x.append(np.max(x))
    max_y.append(np.max(y))
    max_z.append(np.max(z))
    max_xyz.append(np.max(xyz))

    # 最小値
    min_x.append(np.min(x))
    min_y.append(np.min(y))
    min_z.append(np.min(z))
    min_xyz.append(np.min(xyz))

    # # 中央値
    # median_x.append(np.median(x))
    # median_y.append(np.median(y))
    # median_z.append(np.median(z))
    # median_xyz.append(np.median(xyz))

    # 範囲
    range_x.append(np.ptp(x))
    range_y.append(np.ptp(y))
    range_z.append(np.ptp(z))
    range_xyz.append(np.ptp(xyz))

    # フーリエ変換
    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)
    fft_z = np.fft.fft(z)
    fft_xyz = np.fft.fft(xyz)

    # パワースペクトル密度
    power_x = np.abs(fft_x) ** 2
    power_y = np.abs(fft_y) ** 2
    power_z = np.abs(fft_z) ** 2
    power_xyz = np.abs(fft_xyz) ** 2
    power_spectrum_x.append(power_x)
    power_spectrum_y.append(power_y)
    power_spectrum_z.append(power_xyz)
    power_spectrum_xyz.append(power_xyz)

    # 周波数成分
    peak_frequency_x.append(np.argmax(power_x))
    peak_frequency_y.append(np.argmax(power_y))
    peak_frequency_z.append(np.argmax(power_z))
    peak_frequency_xyz.append(np.argmax(power_xyz))

    # 周波数帯域のエネルギー
    peak_amplitude_x.append(np.max(power_x))
    peak_amplitude_y.append(np.max(power_y))
    peak_amplitude_z.append(np.max(power_z))
    peak_amplitude_xyz.append(np.max(power_xyz))

    # # 自己相関関数
    # auto_corr_x.append(np.correlate(x, x, mode='full'))
    # auto_corr_y.append(np.correlate(y, y, mode='full'))
    # auto_corr_z.append(np.correlate(z, z, mode='full'))
    # auto_corr_xyz.append(np.correlate(xyz, xyz, mode='full'))

    # エネルギーベースの特徴量
    # integral_energy_x.append(np.trapz(x ** 2))
    # integral_energy_y.append(np.trapz(y ** 2))
    # integral_energy_z.append(np.trapz(z ** 2))
    # integral_energy_xyz.append(np.trapz(xyz ** 2))





