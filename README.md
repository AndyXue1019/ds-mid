# NCUMA Data Science Midterm

### Team members: [薛安佑](https://github.com/AndyXue1019), [馬沅辰](https://github.com/Ama0124)

## How to run this project:
### 1. Clone this repository into your ROS(1) workspace:
```bash
cd ~/catkin_ws/src
git clone https://github.com/AndyXue1019/ds-mid.git
```

### 2. Build and source your catkin workspace:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 3. Run the file:
- ### Labeling tool:
    ```bash
    roslaunch ds_mid labeling.launch
    ```
- ### Main (Tracking ball and box)
    #### Change the <ROBOT_NAME> into the bot you are using.
    ```bash
    roslaunch ds_mid main.launch robot:=<ROBOT_NAME>
    ```
