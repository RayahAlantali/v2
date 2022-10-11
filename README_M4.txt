Steps to remember when marking:
1. Download map and search list
2. Place map in catkin_ws/src/penguinpi_gazebo using Ubuntu>home>charl>... manual copy and paste (may need to use dos2unix)
3. May need to rename map to M4_true_map_3fruits.txt

4. Download code repo and place in Ubuntu>home>charl and rename as LiveDemo
5. Place search_list in main folder
6. If running on actual robot, replace param folder

7. source ~/LiveDemo/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch

8. source ~/LiveDemo/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l M4_true_map_3fruits.txt

9. python3 AFR2.py
10. Press g to run 