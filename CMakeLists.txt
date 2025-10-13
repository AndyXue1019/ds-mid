cmake_minimum_required(VERSION 3.0.2)
project(ds_mid)

find_package(catkin REQUIRED
  rospy
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package(CATKIN_DEPENDS
  rospy
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
