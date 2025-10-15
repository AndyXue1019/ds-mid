#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import colorsys
import os
import time

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.preprocessing import StandardScaler

from utils.Adaboost import adaboost_predict
from utils.KalmanFilter import KalmanFilter as KF
from utils.Segment import extract_features, merge_segments, segment

ROBOT_NAME = 'turtlebot'
marker_pub = None
i = 0
stumps = None
alphas = None
scaler = None

trackers = {}
next_tracker_id = 0
last_scan_time = None

LABEL_COLORS = {
    'B': (0.0, 0.0, 1.0),  # Blue for Box
    'C': (1.0, 0.0, 0.0),  # Red for Circle
    'O': (0.5, 0.5, 0.5),  # Gray for Other
}


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


def marker_publish(tracked_objects, frame_id):
    marker_array = MarkerArray()

    # 清除舊的 Markers
    clear_marker = Marker()
    clear_marker.header.frame_id = frame_id
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    marker_pub.publish(marker_array)

    marker_array = MarkerArray()

    for obj_id, obj_data in tracked_objects.items():
        # 從追蹤器獲取平滑後的位置
        x, y = obj_data['kf'].x[0, 0], obj_data['kf'].x[1, 0]
        label = obj_data['label']
        score = obj_data['score']

        # 點的 Marker
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'tracked_objects'
        marker.id = obj_id * 2  # 每個物體有2個marker，ID不能重複
        marker.type = Marker.SPHERE if label == 'C' else Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        r, g, b = LABEL_COLORS.get(label, (1.0, 1.0, 1.0))
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, 1.0
        marker_array.markers.append(marker)

        # 文字 Marker (顯示 ID 和分數)
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = 'object_labels'
        text_marker.id = obj_id * 2 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = 0.3  # 文字在物體上方
        text_marker.text = (
            'Ball\n' if label == 'C' else 'Box\n'
        ) + f'Score:{score:.2f}'
        text_marker.scale.z = 0.15  # 文字大小
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        marker_array.markers.append(text_marker)

    if marker_array.markers:
        marker_pub.publish(marker_array)


def scan_callback(scan: LaserScan):
    global i, stumps, alphas, scaler, trackers, next_tracker_id, last_scan_time
    if stumps is None or alphas is None or scaler is None:
        rospy.logwarn('模型或Scaler尚未載入。')
        return

    # 計算時間差 dt
    current_time = time.time()
    if last_scan_time is None:
        last_scan_time = current_time
        return
    dt = current_time - last_scan_time
    last_scan_time = current_time

    # 1. 預測步驟: 對所有現有追蹤器進行預測
    for _, tracker_data in trackers.items():
        tracker_data['kf'].predict(dt)

    # 2. 偵測步驟: 從 LaserScan 獲取當前幀的偵測結果
    ranges = np.array(scan.ranges)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.vstack((x, y)).T  # N x 2 array

    # 呼叫分割函數
    seg_orig, _, _ = segment(points)
    Seg, Si_n, S_n = merge_segments(seg_orig, points)

    detections = []  # 儲存本幀所有的偵測結果
    for seg_idx in range(S_n):
        if Si_n[seg_idx] < 3:
            continue

        segment_points = points[Seg[seg_idx]]

        features = extract_features(segment_points)
        features_scaled = scaler.transform(np.array([features]))

        predicted_label = adaboost_predict(features_scaled, stumps, alphas)[0]

        # 只關心 Ball ('C') 和 Box ('B')
        if predicted_label in ['C', 'B']:
            center = np.mean(segment_points, axis=0)
            detections.append({'label': predicted_label, 'center': center})

    # 3. 資料關聯與更新步驟
    association_dist_thresh = 0.8
    matched_detections = [False] * len(detections)

    # 嘗試將每個追蹤器與最近的偵測結果匹配
    for _, tracker_data in trackers.items():
        predicted_pos = tracker_data['kf'].x[:2].flatten()
        best_dist = float('inf')
        best_det_idx = -1

        for det_idx, det in enumerate(detections):
            # 如果偵測結果已匹配，或者標籤不符，則跳過
            if matched_detections[det_idx] or det['label'] != tracker_data['label']:
                continue

            dist = np.linalg.norm(predicted_pos - det['center'])
            if dist < association_dist_thresh and dist < best_dist:
                best_dist = dist
                best_det_idx = det_idx

        if best_det_idx != -1:
            tracker_data['kf'].update(detections[best_det_idx]['center'])
            matched_detections[best_det_idx] = True
            tracker_data['unmatched_frames'] = 0

            # 生命週期確認
            if tracker_data['status'] == 'tentative':
                tracker_data['hits'] += 1
                if tracker_data['hits'] >= 3:  # 連續命中3次，確認
                    tracker_data['status'] = 'confirmed'
        else:
            # 如果沒找到匹配
            tracker_data['unmatched_frames'] += 1

    # 4. 追蹤器生命週期管理
    # 創建新的'暫定'追蹤器
    for det_idx, det in enumerate(detections):
        if not matched_detections[det_idx]:
            new_kf = KF(x_init=det['center'][0], y_init=det['center'][1], dt=dt)
            trackers[next_tracker_id] = {
                'kf': new_kf,
                'label': det['label'],
                'unmatched_frames': 0,
                'id': next_tracker_id,
                'status': 'tentative',
                'hits': 1,
            }
            next_tracker_id += 1

    # 刪除舊的追蹤器 (連續多幀未匹配的)
    ids_to_delete = []
    max_unmatched_frames = 10
    for tracker_id, tracker_data in trackers.items():
        # confirmed 的追蹤器如果失配太久，就刪除
        if (
            tracker_data['status'] == 'confirmed'
            and tracker_data['unmatched_frames'] > max_unmatched_frames
        ):
            ids_to_delete.append(tracker_id)
        # tentative 的追蹤器只要失配一次，就刪除
        elif (
            tracker_data['status'] == 'tentative'
            and tracker_data['unmatched_frames'] > 0
        ):
            ids_to_delete.append(tracker_id)

    for tracker_id in ids_to_delete:
        del trackers[tracker_id]

    # 5. 準備輸出與可視化
    tracked_objects_for_viz = {}

    rospy.loginfo(f'i: {i}, 追蹤器數量: {len(trackers)}')
    for tracker_id, tracker_data in trackers.items():
        # 只處理 confirmed 的追蹤器
        if tracker_data['status'] != 'confirmed':
            continue

        pos_uncertainty = np.trace(tracker_data['kf'].P[:2, :2])

        score = max(0, 80 - pos_uncertainty * 20)

        tracked_objects_for_viz[tracker_id] = {
            'kf': tracker_data['kf'],
            'label': tracker_data['label'],
            'score': score,
        }
        # (object_x, object_y, probability, object_index)
        rospy.loginfo(
            f'{"Ball " if tracker_data["label"] == "C" else "Box  "} '
            f'x={tracker_data["kf"].x[0, 0]:.2f}, '
            f'y={tracker_data["kf"].x[1, 0]:.2f}, '
            f'score={score:.2f}, '
            f"label='{tracker_data['label']}'"
        )

    # rviz標記
    marker_publish(tracked_objects_for_viz, scan.header.frame_id)

    i += 1


def main():
    global marker_pub, stumps, alphas, scaler, last_scan_time
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

    last_scan_time = time.time()

    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()


if __name__ == '__main__':
    main()
