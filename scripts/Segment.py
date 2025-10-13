#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple


def segment(xy: np.ndarray) -> Tuple[List[List[int]], List[int], int]:
    """
    將 2D 雷射掃描點分割成數個連續的片段。

    :param xy: 一個 N x 2 的 NumPy 陣列，代表雷射點的 (x, y) 座標。
    :type xy: np.ndarray

    :return: 回傳tuple(segments, segment_sizes, num_segments): \n
        - segments (list of list of int): 每個子列表包含一個片段的點索引 (0-based)\n
        - segment_sizes (list of int): 每個片段的點數量\n
        - num_segments (int): 總片段數量\n
    :rtype: tuple[list[list[int]], list[int], int]
    """
    # 定義距離閾值 (單位：公尺)
    threshold = 0.3

    # 找出所有不是 (0,0) 的有效點的 0-based 索引
    # np.any(xy != 0, axis=1) 會回傳一個布林陣列，標示哪些行不全是零
    # np.where(...) 會找出這些布林值為 True 的索引
    nonzero_indices = np.where(np.any(xy != 0, axis=1))[0]

    # 如果沒有任何有效點，直接回傳空值
    if len(nonzero_indices) == 0:
        return [], [], 0

    # 初始化結果列表
    # segments 是一個列表的列表，用來儲存所有片段
    # 將第一個有效點的索引作為第一個片段的起點
    segments = [[nonzero_indices[0]]]

    # 遍歷剩下的有效點 (從第二個開始)
    for i in range(1, len(nonzero_indices)):
        # 獲取當前點和前一個點在原始 xy 陣列中的索引
        current_idx = nonzero_indices[i]
        prev_idx = nonzero_indices[i - 1]

        # 獲取點的實際座標
        current_point = xy[current_idx]
        prev_point = xy[prev_idx]

        # 計算兩點之間的歐幾里得距離
        distance = np.linalg.norm(current_point - prev_point)

        if distance < threshold:
            # --- 距離小於閾值：屬於同一個片段 ---
            # 將當前點的索引加入到最後一個 (也就是當前的) 片段中
            segments[-1].append(current_idx)
        else:
            # --- 距離大於閾值：開始一個新片段 ---
            # 新增一個只包含當前點索引的列表到 segments 中
            segments.append([current_idx])

    # 計算每個片段的大小
    segment_sizes = [len(seg) for seg in segments]

    # 總片段數
    num_segments = len(segments)

    return segments, segment_sizes, num_segments
