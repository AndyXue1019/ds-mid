#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import colorsys
import os

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from utils.Adaboost import adaboost_predict, adaboost_predict_proba
from utils.BayesFilter import BayesFilter
from utils.Segment import extract_features, merge_segments, segment

ROBOT_NAME = 'turtlebot'
marker_pub = None
i = 0
stumps = None
alphas = None
scaler = None

tracked_objects = {}
next_object_id = 0
 
LABEL_COLORS = {
    2 : (0.0, 0.0, 1.0),  # Blue for Box
    1 : (1.0, 0.0, 0.0),  # Red for Ball
    0 : (0.5, 0.5, 0.5),  # Gray for Other
}


class ObjectTracker:
    """一個簡單的物件追蹤器類別"""
    def __init__(self, object_id, initial_points, initial_observation_proba):
        self.id = object_id
        self.centroid = np.mean(initial_points, axis=0)
        # 初始信念可以基於第一次的觀測
        self.filter = BayesFilter(initial_belief=initial_observation_proba)
        self.time_since_update = 0
        self.history = [self.centroid] # 記錄歷史位置

def gen_hsv_colors(i, total):
    """
    根據索引生成一個鮮豔的 HSV 顏色，並轉換為 RGB。
    """
    hue = float(i) / total
    saturation = 1.0
    value = 1.0
    # colorsys.hsv_to_rgb 回傳的是 0-1 範圍的 float
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return r, g, b


def marker_publish(segments, points, predictions, frame_id):
    marker_array = MarkerArray()

    clear_marker = Marker()
    clear_marker.header.frame_id = frame_id
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    marker_pub.publish(marker_array)

    marker_array = MarkerArray()
    object_id_counter = 0

    for i, seg in enumerate(segments):
        if not seg:
            continue

        pred_label = predictions[i]
        if pred_label == 0:
            continue  # 不顯示 Other 類別

        # 計算標記的位置 (使用 segment 的質心)
        segment_points = points[seg]
        centroid = np.mean(segment_points, axis=0)

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'object_markers'
        marker.id = object_id_counter
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0  # 無旋轉

        # 設定標記的中心位置
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = 0.1  # 將標記稍微抬高，避免與地面重疊

        # 根據預測的標籤設定 Marker 的類型、大小和顏色
        if pred_label == 1:  # Ball -> SPHERE
            marker.type = Marker.SPHERE
            
            # 根據 segment 點的分佈來估計球的直徑
            marker.scale.x = 0.2 # 設定最小直徑
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            # 從 LABEL_COLORS 字典獲取顏色
            r, g, b = LABEL_COLORS.get(1, (1.0, 0.0, 0.0)) # 預設紅色
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.8 # 稍微透明

        elif pred_label == 2:  # Box -> CUBE
            marker.type = Marker.CUBE

            # 根據 segment 點在 x, y 軸上的跨度來估計箱子的大小
            marker.scale.x = 0.15  # 設定最小寬度
            marker.scale.y = 0.15  # 設定最小長度
            marker.scale.z = 0.15  # 設定固定高度

            # 從 LABEL_COLORS 字典獲取顏色
            r, g, b = LABEL_COLORS.get(2, (0.0, 0.0, 1.0)) # 預設藍色
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.8 # 稍微透明

        # 將設定好的 marker 加入到 array 中
        marker_array.markers.append(marker)
        object_id_counter += 1

    if marker_array.markers:
        marker_pub.publish(marker_array)


def scan_callback(scan: LaserScan):
    global i, stumps, alphas, scaler, tracked_objects, next_object_id
    if stumps is None or alphas is None or scaler is None:
        rospy.logwarn('模型或標準化參數尚未載入。')
        return

    ranges = np.array(scan.ranges)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

    # 將極座標轉換為直角座標
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.vstack((x, y)).T  # N x 2 array

    # 呼叫分割函數
    seg_orig, _, _ = segment(points)
    Seg, Si_n, S_n = merge_segments(seg_orig, points)

    # === 1. 對當前所有 segments 進行分類預測 ===
    current_detections = []
    features_to_predict = []
    idx_to_process = []
    for seg_idx in range(S_n):
        if Si_n[seg_idx] < 3:
            continue
        segment_points = points[Seg[seg_idx]]
        features = extract_features(segment_points)
        features_to_predict.append(features)
        idx_to_process.append(seg_idx)
    
    if not features_to_predict:
        # 如果沒有偵測到任何東西，更新所有追蹤物件的消失計時器
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id].time_since_update += 1
            if tracked_objects[obj_id].time_since_update > 10: # 例如超過10幀沒看到就刪除
                del tracked_objects[obj_id]
        marker_publish([], points, [], scan.header.frame_id) # 清除標記
        return

    features_array = np.array(features_to_predict)
    features_scaled = scaler.transform(features_array)
    # 得到機率分佈
    probabilities, classes = adaboost_predict_proba(features_scaled, stumps, alphas)

    for idx, seg_idx in enumerate(idx_to_process):
        current_detections.append({
            'centroid': np.mean(points[Seg[seg_idx]], axis=0),
            'points_indices': Seg[seg_idx],
            'probabilities': probabilities[idx]
        })

    # === 2. 資料關聯 (Data Association) ===
    if not tracked_objects:
        # 如果還沒有任何追蹤物件，將所有當前偵測設為新物件
        for det in current_detections:
            tracked_objects[next_object_id] = ObjectTracker(next_object_id, points[det['points_indices']], det['probabilities'])
            next_object_id += 1
    else:
        # 計算 cost matrix (距離)
        tracker_ids = list(tracked_objects.keys())
        tracker_centroids = np.array([tracked_objects[tid].centroid for tid in tracker_ids])
        detection_centroids = np.array([det['centroid'] for det in current_detections])
        
        cost_matrix = np.linalg.norm(tracker_centroids[:, np.newaxis, :] - detection_centroids[np.newaxis, :, :], axis=2)

        # 使用匈牙利演算法找到最佳配對
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_indices = []
        dist_threshold = 0.5 # 超過此距離的配對無效
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < dist_threshold:
                matched_indices.append((r, c))

        # === 3. 更新、新增、刪除 ===
        matched_tracker_indices = {r for r, c in matched_indices}
        matched_detection_indices = {c for r, c in matched_indices}

        # 更新配對成功的追蹤器
        for r, c in matched_indices:
            tracker_id = tracker_ids[r]
            detection = current_detections[c]
            
            tracked_objects[tracker_id].filter.update(detection['probabilities'])
            tracked_objects[tracker_id].centroid = detection['centroid']
            tracked_objects[tracker_id].time_since_update = 0

        # 為未配對的偵測建立新追蹤器
        for c in range(len(current_detections)):
            if c not in matched_detection_indices:
                detection = current_detections[c]
                tracked_objects[next_object_id] = ObjectTracker(next_object_id, points[detection['points_indices']], detection['probabilities'])
                next_object_id += 1

        # 處理未配對的追蹤器 (可能已消失)
        for r in range(len(tracker_ids)):
            if r not in matched_tracker_indices:
                tracker_id = tracker_ids[r]
                tracked_objects[tracker_id].time_since_update += 1
                if tracked_objects[tracker_id].time_since_update > 10:
                    del tracked_objects[tracker_id]
                    
    # === 4. 準備發布與日誌紀錄 ===
    final_segments_to_publish = []
    final_predictions_to_publish = []
    log_output = []
    class_names = {0: 'Other', 1: 'Ball', 2: 'Box'}

    for obj_id, tracker in tracked_objects.items():
        # 從濾波後的信念中找出最可能的類別
        final_belief = tracker.filter.belief
        best_class_idx = np.argmax(final_belief)
        best_class_prob = final_belief[best_class_idx]
        
        # 找到這個物件最新的 segment 點
        # 這裡需要一個方法來重新找到 tracker 對應的 segment 點
        # 為簡化，我們可以在 tracker 中儲存最新的 segment 點
        # 這裡我們先用一個簡化邏輯
        # 找到離 tracker centroid 最近的 detection
        if current_detections:
            dists = [np.linalg.norm(tracker.centroid - det['centroid']) for det in current_detections]
            closest_det_idx = np.argmin(dists)
            if dists[closest_det_idx] < 0.5: # 確保是同一個物體
                final_segments_to_publish.append(current_detections[closest_det_idx]['points_indices'])
                final_predictions_to_publish.append(best_class_idx)

        if best_class_idx in [1, 2]:
            best_class_prob = final_belief[best_class_idx]
            obj_x = tracker.centroid[0]
            obj_y = tracker.centroid[1]
            # 依照您指定的格式 (object_x, object_y, probability, 1 or 2) 建立字串
            log_output.append(f'(x={obj_x:.2f}, y={obj_y:.2f}, probability={best_class_prob:.4f}, index={best_class_idx})')

    if log_output:
        print('-' * 45)
        print('\n'.join(log_output))
    marker_publish(final_segments_to_publish, points, final_predictions_to_publish, scan.header.frame_id)
    
    i += 1


def main():
    global marker_pub, stumps, alphas, scaler
    rospy.init_node('ds_mid_node')

    model_path = os.path.join(
        os.path.dirname(__file__), f'../model/{ROBOT_NAME}/adaboost_model.npz'
    )
    try:
        model = np.load(model_path, allow_pickle=True)
        stumps = model['stumps']
        alphas = model['alphas']
        rospy.loginfo('模型載入成功。')
    except FileNotFoundError:
        rospy.logerr(f'找不到模型檔案: {model_path}')
        return

    scaler_path = os.path.join(
        os.path.dirname(__file__), f'../model/{ROBOT_NAME}/scaler.npz'
    )
    try:
        scaler_data = np.load(scaler_path)
        scaler = StandardScaler()
        scaler.mean_ = scaler_data['mean']
        scaler.scale_ = scaler_data['scale']
        rospy.loginfo('Scaler載入成功。')
    except FileNotFoundError:
        rospy.logerr(f'找不到Scaler檔案: {scaler_path}')
        return

    marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)

    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()


if __name__ == '__main__':
    main()