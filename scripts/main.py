#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import colorsys
import os

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.preprocessing import StandardScaler

from utils.Adaboost import adaboost_predict
from utils.Segment import extract_features, merge_segments, segment

ROBOT_NAME = 'turtlebot'
marker_pub = None
i = 0
stumps = None
alphas = None
scaler = None

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


def marker_publish(segments, points, predictions, frame_id):
    marker_array = MarkerArray()

    clear_marker = Marker()
    clear_marker.header.frame_id = frame_id
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    marker_pub.publish(marker_array)

    marker_array = MarkerArray()

    for i, seg in enumerate(segments):
        if not seg:
            continue

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()

        marker.ns = 'laser_segments'
        marker.id = i + 1  # ID 從 1 開始，0 保留給 DELETEALL

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 點的大小
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.05  # 不重要

        # 點的顏色
        pred_label = predictions[i]
        r, g, b = LABEL_COLORS.get(pred_label, (1.0, 1.0, 1.0))  # 預設為白色
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0  # 不透明

        marker.points = [Point(points[idx, 0], points[idx, 1], 0) for idx in seg]

        marker_array.markers.append(marker)

        if seg:
            seg_points = points[seg]
            coords = np.mean(seg_points, axis=0)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = rospy.Time.now()

            text_marker.ns = 'segment_labels'
            text_marker.id = i + 1

            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = coords[0]
            text_marker.pose.position.y = coords[1]
            text_marker.pose.position.z = 0.2  # 文字稍微抬高
            text_marker.pose.orientation.w = 1.0  # 無旋轉

            text_marker.text = pred_label  # 文字內容
            text_marker.scale.z = 0.2  # 文字高度

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0  # 不透明

            marker_array.markers.append(text_marker)

    if marker_array.markers:
        marker_pub.publish(marker_array)


def scan_callback(scan: LaserScan):
    global i, stumps, alphas, scaler
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

    # rospy.loginfo(f'{i}: Found {num_segments} segments.')

    predictions = [''] * S_n
    features_to_predict = []
    idx_to_predict = []

    for seg_idx in range(S_n):
        if Si_n[seg_idx] < 3:
            predictions[seg_idx] = 'O'  # 點太少，直接標為'Other'
            continue

        segment_points = points[Seg[seg_idx]]
        # 提取特徵，結構必須與訓練時完全相同
        features = extract_features(segment_points)
        features_to_predict.append(features)
        idx_to_predict.append(seg_idx)

    # 一次性對所有需要預測的片段進行預測
    if features_to_predict:
        features_array = np.array(features_to_predict)
        features_scaled = scaler.transform(features_array)
        predicted_labels = adaboost_predict(features_scaled, stumps, alphas)
        for idx, label in zip(idx_to_predict, predicted_labels):
            predictions[idx] = label

    rospy.loginfo(f'{i}: Found {S_n} segments.\nPredictions: {predictions}')

    # rviz標記
    marker_publish(Seg, points, predictions, scan.header.frame_id)

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