#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import os

import matplotlib.pyplot as plt
import rospkg
from matplotlib.widgets import Button, RadioButtons

from utils.Segment import handle_scan
from utils.Loader import load_bag_data, load_label

os.chdir(rospkg.RosPack().get_path('ds_mid'))

COLOR_BALL = 'red'
COLOR_BOX = 'blue'
COLOR_OTHER = 'gray'


class LabelingTool:
    def __init__(self, scan_msgs, label_file):
        self.scan_msgs = scan_msgs
        self.label_file = label_file
        self.labels = self.load_existing_labels()

        self.current_scan_index = 0
        self.current_label_mode = 'ball'  # 預設標記模式

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.25, bottom=0.2)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.59, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, 'Prev (←)')
        self.btn_next = Button(ax_next, 'Next (→)')
        self.btn_save = Button(ax_save, 'Save (s)')
        self.btn_prev.on_clicked(self.prev_scan)
        self.btn_next.on_clicked(self.next_scan)
        self.btn_save.on_clicked(self.save_button_callback)

        ax_radio = plt.axes([0.05, 0.7, 0.15, 0.15])
        self.radio_buttons = RadioButtons(ax_radio, ('ball', 'box'))
        self.radio_buttons.on_clicked(self.set_label_mode)

        print('\n--- 操作說明 ---')
        print("點擊 'ball' 或 'box' 切換標記模式。")
        print('點擊畫面中的 Segment 進行標記。')
        print("按 '←' 或 '→' (或點擊按鈕) 切換畫面。")
        print("按 's' (或點擊按鈕) 儲存標籤。")
        print("按 'c' 清除當前畫面的標籤。")
        print("按 'q' 關閉視窗。")

        self.plot_scan()

    def load_existing_labels(self):
        labels = [{} for _ in self.scan_msgs]
        if os.path.exists(self.label_file):
            loaded_labels = load_label(self.label_file)
            for i, label_dict in enumerate(loaded_labels):
                labels[i] = label_dict
        print(f'Loaded existing labels from {self.label_file}')
        return labels

    def set_label_mode(self, label):
        self.current_label_mode = label
        print(f'Switching label mode to: {label}')

    def plot_scan(self):
        self.ax.clear()
        scan = self.scan_msgs[self.current_scan_index]
        Seg, _, S_n, points = handle_scan(scan)

        current_labels = self.labels[self.current_scan_index]
        ball_seg_idx = current_labels.get('ball', None)
        box_seg_idx = current_labels.get('box', None)

        if (ball_seg_idx is not None) and (box_seg_idx is None):
            # 只有 ball，預設切換到 box
            new_mode_index = 1  # 'box' 的索引是 1
            self.current_label_mode = 'box'
        else:
            # 其他所有情況 (只有 box, 兩個都有, 兩個都無)
            # 預設都是 'ball'
            new_mode_index = 0  # 'ball' 的索引是 0
            self.current_label_mode = 'ball'

        self.radio_buttons.set_active(new_mode_index)

        for i in range(S_n):
            seg_points = points[Seg[i]]

            if i == ball_seg_idx:
                color = COLOR_BALL
            elif i == box_seg_idx:
                color = COLOR_BOX
            else:
                color = COLOR_OTHER

            self.ax.scatter(
                seg_points[:, 0],
                seg_points[:, 1],
                s=1,
                color=color,
                label=str(i),
                picker=True,
                pickradius=5,
            )

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        title = f'Scan {self.current_scan_index + 1}/{len(self.scan_msgs)}\n'
        title += f'Ball: {ball_seg_idx if ball_seg_idx is not None else "---"} | '
        title += f'Box: {box_seg_idx if box_seg_idx is not None else "---"}'
        self.ax.set_title(title)

        plt.draw()

    def on_pick(self, event):
        artist = event.artist
        seg_idx_str = artist.get_label()

        try:
            seg_idx = int(seg_idx_str)
        except ValueError:
            return

        mode = self.current_label_mode
        current_labels = self.labels[self.current_scan_index]

        other_mode = 'box' if mode == 'ball' else 'ball'
        if current_labels.get(other_mode, None) == seg_idx:
            current_labels.pop(other_mode)
            print(f'Removed {other_mode} label from segment {seg_idx}')

        current_labels[mode] = seg_idx
        print(
            f'--- Frame {self.current_scan_index}: Labeled segment {seg_idx} as {mode} ---'
        )

        self.plot_scan()

    def on_key(self, event):
        if event.key == 'right':
            self.next_scan(None)
        elif event.key == 'left':
            self.prev_scan(None)
        elif event.key == 's':
            self.save_labels()
        elif event.key == 'c':
            # 清除當前畫面的標籤
            self.labels[self.current_scan_index].clear()
            print(f'--- Frame {self.current_scan_index}: Cleared labels ---')
            self.plot_scan()
        elif event.key == 'q':
            plt.close(self.fig)

    def next_scan(self, _):
        if self.current_scan_index < len(self.scan_msgs) - 1:
            self.current_scan_index += 1
            self.plot_scan()

    def prev_scan(self, _):
        if self.current_scan_index > 0:
            self.current_scan_index -= 1
            self.plot_scan()

    def save_button_callback(self, _):
        self.save_labels()

    def save_labels(self):
        fieldnames = ['ball', 'box']

        row = []
        for label_dict in self.labels:
            if label_dict:
                row.append(
                    {
                        'ball': label_dict.get('ball', ''),
                        'box': label_dict.get('box', ''),
                    }
                )

        try:
            with open(self.label_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(row)
            print(f'Labels saved to {self.label_file}')
        except IOError as e:
            print(f'Error saving labels to {self.label_file}: {e}')


def main():
    parser = argparse.ArgumentParser(description='Labeling Tool')
    parser.add_argument(
        'bag_file', type=str, help='Path to the ROS bag file containing LaserScan data'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='(Optional) Path to save the labeled data (CSV format)',
    )
    args = parser.parse_args()

    bag_file_path = args.bag_file
    if args.output:
        label_file_path = args.output
    else:
        label_file_path = bag_file_path.replace('.bag', '_label.csv')
    print(f'Bag file to label: {bag_file_path}')
    print(f'Label file will be saved to: {label_file_path}')

    scan_msgs = load_bag_data(bag_file_path)
    if not scan_msgs:
        print('No LaserScan messages found in the bag file.')
        return

    _ = LabelingTool(scan_msgs, label_file_path)
    plt.show()


if __name__ == '__main__':
    main()
