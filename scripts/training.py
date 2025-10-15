#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
from typing import List

import numpy as np
import rosbag
from sensor_msgs.msg import LaserScan
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.Adaboost import adaboost_predict, adaboost_train
from utils.Segment import extract_features, merge_segments, segment

ROBOT_NAME = 'turtlebot'
data_full = []
label_full = []


def load_labels(data_file):
    ball = []
    box = []
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標題列
        for x, y in reader:
            ball.append(x)
            box.append(y)
    return ball, box


def load_bag_data(bag_file) -> List[LaserScan]:
    """
    從 ROS bag 檔案中讀取 LaserScan 資料。
    """
    bag = rosbag.Bag(bag_file)
    scan_msgs = []
    for _, msg, _ in bag.read_messages(topics=['/scan']):
        scan_msgs.append(msg)
    bag.close()
    print(f'{bag_file}: 讀取到 {len(scan_msgs)} 筆 LaserScan 訊息。')
    return scan_msgs


def label_segments(t, num_segments, data_file):
    ball, box = load_labels(data_file)

    PN = ['O'] * num_segments  # 預設標籤為 'O' (Other)
    PN[int(box[t])] = 'B'
    PN[int(ball[t])] = 'C'

    return PN


def data_label_prepare(bag_file, label_file):
    global data_full, label_full

    scan_msgs = load_bag_data(bag_file)

    for t, scan in enumerate(scan_msgs):
        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T  # N x 2 array

        Seg, _, _ = segment(points)
        Seg, Si_n, S_n = merge_segments(Seg, points)

        PN = label_segments(t, S_n, label_file)

        # # 提取每個片段的特徵
        for i in range(S_n):
            if Si_n[i] < 3:
                continue
            segment_points = np.array([points[idx] for idx in Seg[i]])

            features = extract_features(segment_points)

            data_full.append(features)
            label_full.append(PN[i])


def main():
    global data_full, label_full
    bag_file = f'./data/{ROBOT_NAME}/data_{{}}.bag'
    label_file = f'./data/{ROBOT_NAME}/data_{{}}_label.csv'

    # 準備訓練資料 (可自訂)
    print('準備資料...')
    for i in [1, 2, 3]:
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
    class_labels = ['B', 'C', 'O']  # Box, Circle, Other

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

        bc_indices = np.where((label_test == 'B') | (label_test == 'C'))
        if len(bc_indices[0]) > 0:
            bc_acc = accuracy_score(label_test[bc_indices], test_pred[bc_indices])
        else:
            bc_acc = 0.0  # 如果測試集中沒有 B 或 C，則準確率為 0
        bc_accuracies.append(bc_acc)

        print(cm)
        print(f'準確率: {acc * 100:.4f}%')
        print(f'B/C 準確率: {bc_acc * 100:.4f}%')

    print('\n=== 整體結果 ===')
    print(total_cm)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f'平均準確率: {mean_acc * 100:.4f}% (+/- {std_acc * 100:.4f}%)')
    mean_bc_acc = np.mean(bc_accuracies)
    std_bc_acc = np.std(bc_accuracies)
    print(f'平均 B/C 準確率: {mean_bc_acc * 100:.4f}% (+/- {std_bc_acc * 100:.4f}%)')

    scaler = StandardScaler()
    data_full_scaled = scaler.fit_transform(data_full)

    stumps, alphas = adaboost_train(data_full_scaled, label_full, T=50)

    # 儲存訓練好的模型和標準化參數
    np.savez(f'./model/{ROBOT_NAME}/adaboost_model.npz', stumps=stumps, alphas=alphas)
    np.savez(f'./model/{ROBOT_NAME}/scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    print(f'模型與標準化參數已儲存至 ./model/{ROBOT_NAME}/ 資料夾。')


if __name__ == '__main__':
    main()
