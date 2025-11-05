#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import rospkg
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.Adaboost import adaboost_predict, adaboost_train
from utils.Segment import extract_features, handle_scan
from utils.Loader import load_bag_data, load_label

os.chdir(rospkg.RosPack().get_path('ds_mid'))

ROBOT_NAME = 'minibot'
data_full = []
label_full = []


def data_label_prepare(bag_file, label_file):
    global data_full, label_full

    scan_msgs = load_bag_data(bag_file)
    labels = load_label(label_file)

    for t, scan in enumerate(scan_msgs):
        Seg, Si_n, S_n, filtered_points = handle_scan(scan)
        PN = [0] * S_n  # 預設標籤為 'O' (Other)
        PN[labels[t]['ball']] = 1
        PN[labels[t]['box']] = 2

        # 提取每個片段的特徵
        for i in range(S_n):
            if Si_n[i] < 3:  # 太小的片段跳過
                continue
            segment_points = np.array([filtered_points[idx] for idx in Seg[i]])

            features = extract_features(segment_points)

            data_full.append(features)
            label_full.append(PN[i])


def main():
    global data_full, label_full
    bag_file = f'./data/{ROBOT_NAME}/data_{{}}.bag'
    label_file = f'./data/{ROBOT_NAME}/data_{{}}_label.csv'

    # 準備訓練資料
    print('準備資料...')
    for i in [1, 2]:  # 可根據需要調整資料集編號
        print(f'處理第 {i} 筆資料...')
        data_label_prepare(bag_file.format(i), label_file.format(i))

    # 將訓練和測試數據轉換為 NumPy 陣列
    data_full = np.array(data_full)
    label_full = np.array(label_full)

    if data_full.size == 0:
        raise ValueError('訓練資料集為空，請檢查資料準備過程。')
    print(f'總訓練資料集大小: {data_full.shape}')

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    bc_accuracies = []
    total_cm = np.zeros((3, 3), dtype=int)
    class_labels = [0, 1, 2]  # Other, Ball, Box

    for fold, (train_index, test_index) in enumerate(skf.split(data_full, label_full)):
        print(f'\n--- 第 {fold + 1}/{n_splits} 摺 ---')
        data_train, data_test = data_full[train_index], data_full[test_index]
        label_train, label_test = label_full[train_index], label_full[test_index]
        print(f'訓練集大小: {data_train.shape}, 測試集大小: {data_test.shape}')

        # 特徵標準化
        scaler = StandardScaler()
        data_train = scaler.fit_transform(data_train)
        data_test = scaler.transform(data_test)

        stumps, alphas = adaboost_train(data_train, label_train, T=50)

        test_pred = adaboost_predict(data_test, stumps, alphas)

        cm = confusion_matrix(label_test, test_pred, labels=class_labels)
        total_cm += cm

        acc = accuracy_score(label_test, test_pred)
        accuracies.append(acc)

        bc_indices = np.where((label_test == 1) | (label_test == 2))
        if len(bc_indices[0]) > 0:
            bc_acc = accuracy_score(label_test[bc_indices], test_pred[bc_indices])
        else:
            bc_acc = 0.0  # 如果測試集中沒有 1 或 2，則準確率為 0
        bc_accuracies.append(bc_acc)

        print(cm)
        print(f'準確率: {acc * 100:.4f}%')
        print(f'球/箱 準確率: {bc_acc * 100:.4f}%')

    print('\n=== 整體結果 ===')
    print(total_cm)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f'平均準確率: {mean_acc * 100:.4f}% (+/- {std_acc * 100:.4f}%)')
    mean_bc_acc = np.mean(bc_accuracies)
    std_bc_acc = np.std(bc_accuracies)
    print(f'平均 球/箱 準確率: {mean_bc_acc * 100:.4f}% (+/- {std_bc_acc * 100:.4f}%)')

    # 轉換成 Bayes Filter 的Sensor Model (取到小數點後第四位)
    print('\n=== Bayes Filter Sensor Model ===')
    sensor_model = np.zeros((3, 3))
    for i in range(3):
        row_sum = np.sum(total_cm[i, :])
        if row_sum > 0:
            for j in range(3):
                sensor_model[i, j] = round(total_cm[i, j] / row_sum, 4)
        else:
            sensor_model[i, :] = 0.0  # 避免除以零
    print(sensor_model)
    print()

    scaler = StandardScaler()
    data_full_scaled = scaler.fit_transform(data_full)

    stumps, alphas = adaboost_train(data_full_scaled, label_full, T=50)

    # 儲存訓練好的模型、標準化參數、Sensor Model
    np.savez(f'./model/{ROBOT_NAME}/adaboost_model.npz', stumps=stumps, alphas=alphas)
    np.savez(f'./model/{ROBOT_NAME}/scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    np.savez(f'./model/{ROBOT_NAME}/sensor_model.npz', sensor_model=sensor_model)
    print(f'模型、Scaler與Sensor Model已儲存至 ./model/{ROBOT_NAME}/')


if __name__ == '__main__':
    main()
