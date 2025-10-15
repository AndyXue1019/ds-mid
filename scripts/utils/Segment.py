#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


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
            # 距離小於閾值：屬於同一個片段
            # 將當前點的索引加入到最後一個 (也就是當前的) 片段中
            segments[-1].append(current_idx)
        else:
            # 距離大於閾值：開始一個新片段
            # 新增一個只包含當前點索引的列表到 segments 中
            segments.append([current_idx])

    # 計算每個片段的大小
    segment_sizes = [len(seg) for seg in segments]

    # 總片段數
    num_segments = len(segments)

    return segments, segment_sizes, num_segments


def merge_segments(
    segments: List[List[int]],
    points: np.ndarray,
    merge_dist_thresh=0.1,
    linearity_thresh=0.001,
    circularity_thresh=0.05,
) -> Tuple[List[List[int]], List[int], int]:
    """
    合併相鄰且相似的片段。\n
    :param segments: 原始片段列表\n
    :param points: 所有點的座標\n
    :param merge_dist_thresh: 相鄰片段端點的最大合併距離\n
    :param linearity_thresh: 用於判斷片段是否為直線的線性度誤差閾值\n
    :param circularity_thresh: 用於判斷合併後片段是否為圓的圓度誤差閾值\n
    :return: 合併後的片段列表\n
    """
    if not segments:
        return [], [], 0

    current_segments = [list(seg) for seg in segments if seg]

    i = 0
    while i < len(current_segments) - 1:
        seg1 = current_segments[i]
        seg2 = current_segments[i + 1]

        # 1. 檢查距離鄰近性
        p_last_of_seg1 = points[seg1[-1]]
        p_first_of_seg2 = points[seg2[0]]
        dist = np.linalg.norm(p_last_of_seg1 - p_first_of_seg2)

        if dist < merge_dist_thresh:
            # 2. 試探性合併
            temp_merged_indices = seg1 + seg2
            temp_merged_points = points[temp_merged_indices]

            if len(temp_merged_points) < 3:
                i += 1
                continue

            # 3. 檢查合併後新片段的形狀
            # 檢查線性度
            pca = PCA(n_components=2).fit(temp_merged_points)
            merged_linearity_err = pca.explained_variance_ratio_[1]

            # 檢查圓度
            center, radius = fit_circle(temp_merged_points)
            if np.isinf(radius):
                merged_circularity_err = float('inf')
            else:
                distances_to_center = np.linalg.norm(
                    temp_merged_points - center, axis=1
                )
                merged_circularity_err = np.std(distances_to_center)

            # 4. 判斷是否確認合併
            # 如果合併後仍然是直線 或 變成了一個圓，則確認合併
            if (
                merged_linearity_err < linearity_thresh
                or merged_circularity_err < circularity_thresh
            ):
                current_segments[i] = temp_merged_indices
                current_segments.pop(i + 1)
                # 停在原地，檢查新的 seg1 是否能與下一個片段合併
                continue

        # 如果不合併，則繼續檢查下一個
        i += 1

    # 更新 Si_n 和 S_n
    merged_si_n = [len(seg) for seg in current_segments]
    merged_s_n = len(current_segments)

    # 新增邏輯：檢查第一個和最後一個片段是否可以合併
    if merged_s_n > 1:
        seg1 = current_segments[0]
        seg_last = current_segments[-1]

        # 1. 檢查距離鄰近性
        p_last_of_seg_last = points[seg_last[-1]]
        p_first_of_seg1 = points[seg1[0]]
        dist = np.linalg.norm(p_last_of_seg_last - p_first_of_seg1)

        if dist < merge_dist_thresh:
            # 2. 試探性合併
            temp_merged_indices = seg_last + seg1
            temp_merged_points = points[temp_merged_indices]

            if len(temp_merged_points) >= 3:
                # 3. 檢查合併後新片段的形狀 (圓度)
                center, radius = fit_circle(temp_merged_points)
                if not np.isinf(radius):
                    distances_to_center = np.linalg.norm(
                        temp_merged_points - center, axis=1
                    )
                    merged_circularity_err = np.std(distances_to_center)

                    # 4. 如果合併後像一個圓，則確認合併
                    if merged_circularity_err < circularity_thresh:
                        current_segments[0] = temp_merged_indices
                        current_segments.pop(-1)
                        # 更新合併後的片段數量和大小
                        merged_si_n = [len(seg) for seg in current_segments]
                        merged_s_n = len(current_segments)

    return current_segments, merged_si_n, merged_s_n


def filter_outliers(
    points: np.ndarray, radius: float = 0.15, min_neighbors: int = 3
) -> np.ndarray:
    """
    使用 KD-Tree 加速的半徑異常點移除法來過濾噪聲點。

    :param points: N x 2 的 NumPy 陣列，代表雷射點的 (x, y) 座標。
    :param radius: 搜尋鄰居的半徑 (單位：公尺)。
    :param min_neighbors: 一個點被視為非噪聲點所需的最小鄰居數 (不包含點自身)。
    :return: 一個 NumPy 陣列，包含被保留下來的點在原始陣列中的索引。
    """
    # 僅處理非 (0,0) 的有效點
    valid_indices = np.where(np.any(points != 0, axis=1))[0]
    if len(valid_indices) < min_neighbors + 1:
        return np.array([], dtype=int)

    valid_points = points[valid_indices]

    # 1. 建立 KD-Tree
    # leaf_size 可以調整，較大的 leaf_size 可能建樹更快，但查詢稍慢
    tree = KDTree(valid_points, leaf_size=10)

    # 2. 查詢每個點在半徑內的鄰居數量
    # tree.query_radius 回傳的是每個點的鄰居索引列表
    # 我們只需要鄰居的數量，所以取其長度
    # +1 是因為查詢結果會包含點自身，所以 min_neighbors 也要加 1
    neighbors_count = tree.query_radius(valid_points, r=radius, count_only=True)

    # 3. 找出鄰居數量足夠的點的索引 (相對於 valid_points)
    # 鄰居數需大於等於 min_neighbors + 1 (因為包含自身)
    non_outlier_local_indices = np.where(neighbors_count >= min_neighbors + 1)[0]

    # 4. 將局部索引映射回原始 points 陣列的索引
    original_indices_to_keep = valid_indices[non_outlier_local_indices]

    return original_indices_to_keep


def fit_circle(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    使用最小二乘法擬合一個圓。
    返回圓心 (xc, yc) 和半徑 r。
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([2 * x, 2 * y, np.ones(len(x))]).T
    b = x**2 + y**2
    # 解 Ac = b
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 處理擬合失敗的情況
    if c[2] + c[0] ** 2 + c[1] ** 2 < 0:
        return (c[0], c[1]), np.inf
    return (c[0], c[1]), np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)


def extract_features(points: np.ndarray) -> list:
    # 特徵 1: 點的數量
    num_points = len(points)

    # 特徵 2: 片段寬度 (第一個點到最後一個點的距離)
    width = np.linalg.norm(points[0] - points[-1])

    # 特徵 3: 線性度 (使用 PCA)
    # PCA 會找到數據變異最大的方向。
    # explained_variance_ratio_[1] 代表垂直於主方向的變異程度。
    # 對於直線，這個值應該非常小。
    pca = PCA(n_components=2)
    pca.fit(points)
    linearity_err = pca.explained_variance_ratio_[1]

    # 特徵 4: 長寬比 (Aspect Ratio)
    # 判斷物體的形狀，圓形或正方形接近1，長條形則遠大於1。
    # 加上一個極小值避免除以零的錯誤。
    var = pca.explained_variance_
    if var[1] < 1e-6:
        aspect_ratio = 1e6  # 如果寬度趨近於零，視為一個極大的長寬比 (類似一條線)
    else:
        aspect_ratio = np.sqrt(var[0]) / np.sqrt(var[1])

    # 特徵 5: 點密度 (Point Density)
    # 反映點的分佈密集程度。
    if width < 1e-6:
        point_density = 1e6  # 如果寬度極小，視為密度極大
    else:
        point_density = num_points / width

    # 特徵 6、7: 圓度 (擬合圓後的誤差) 、擬合圓的半徑
    # 計算所有點到擬合圓心的距離，然後取其標準差。
    # 對於圓弧，這個標準差應該很小。
    center, radius = fit_circle(points)
    if np.isinf(radius):  # 如果擬合失敗
        circularity_err = 1.0  # 給一個較大的誤差值
        radius = 100.0  # 給一個較大的半徑值
    else:
        distances_to_center = np.linalg.norm(points - center, axis=1)
        circularity_err = np.std(distances_to_center)

    # 特徵 8: 點到質心的距離標準差
    centroid = np.mean(points, axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    std_dev_dist = np.std(distances_to_centroid)

    # 特徵 9: 曲率估計
    # 擬合二次多項式 y = ax^2 + bx + c 來估計曲率
    # 為了旋轉不變性，我們先將點對齊到主軸
    points_transformed = pca.transform(points)
    x_transformed = points_transformed[:, 0]
    y_transformed = points_transformed[:, 1]
    # 擬合二次多項式，曲率約等於 |2a|
    # 檢查x的範圍以避免 "RankWarning"
    if np.max(x_transformed) - np.min(x_transformed) < 1e-4:
        curvature = 0.0
    else:
        poly_coeffs = np.polyfit(x_transformed, y_transformed, 2)
        curvature = np.abs(2 * poly_coeffs[0])

    # 特徵 10: 角度變化的標準差
    angles = []
    if num_points > 2:
        for i in range(num_points - 2):
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2]
            v1 = p1 - p2
            v2 = p3 - p2
            # 計算向量夾角
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angles.append(angle)

    if angles:
        std_angle = np.std(angles)
    else:
        std_angle = 0.0

    return [
        num_points,
        width,
        linearity_err,
        aspect_ratio,
        point_density,
        circularity_err,
        std_dev_dist,
        curvature,
        radius,
        std_angle,
    ]
