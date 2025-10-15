#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import colorsys

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from utils.Segment import filter_outliers, merge_segments, segment

marker_pub = None
i = 0

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


def marker_publish(segments, num_segments, points, frame_id):
    marker_array = MarkerArray()

    clear_marker = Marker()
    clear_marker.header.frame_id = frame_id
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    marker_pub.publish(marker_array)

    marker_array = MarkerArray()

    for i, seg in enumerate(segments):
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
        marker.scale.z = 0.05 # 不重要

        # 點的顏色
        r, g, b = gen_hsv_colors(i, num_segments)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0 # 不透明

        marker.points = [
            Point(points[idx, 0], points[idx, 1], 0)
            for idx in seg
        ]

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
            text_marker.pose.position.z = 0.2 # 文字稍微抬高
            text_marker.pose.orientation.w = 1.0 # 無旋轉

            text_marker.text = str(i) # 文字內容
            text_marker.scale.z = 0.2 # 文字高度

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0 # 不透明

            marker_array.markers.append(text_marker)

    if marker_array.markers:
        marker_pub.publish(marker_array)


def scan_callback(scan: LaserScan):
    global i
    ranges = np.array(scan.ranges)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

    # 將極座標轉換為直角座標
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.vstack((x, y)).T # N x 2 array

    # 過濾
    keep_indices = filter_outliers(points, radius=0.15, min_neighbors=5)
    filtered_points = points[keep_indices]

    # 呼叫分割函數
    Seg, _, _ = segment(filtered_points)
    Seg, _, S_n = merge_segments(Seg, filtered_points)
    rospy.loginfo(f'{i}: Found {S_n} segments.')

    # rviz標記
    marker_publish(Seg, S_n, filtered_points, scan.header.frame_id)

    i += 1


def main():
    global marker_pub
    rospy.init_node('ds_mid_node')

    marker_pub = rospy.Publisher('/laser_segments', MarkerArray, queue_size=10)

    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
