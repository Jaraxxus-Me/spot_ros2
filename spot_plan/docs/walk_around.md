# walk_around
This is a simple example of using ROS 2 to make the robot navigate in a predefined map.

## Running the Example
For this example, make sure the Spot initially can see an april tag that has been used to record the graph_map data, the april tag will be used as the world origin.
```bash
ros2 run spot_plan walk_around
```
If you launched the driver with a namespace, use the following command instead:
```bash
ros2 run spot_plan walk_around --robot <spot_name>
```
The robot should navigate to `[SE2Pose(1.8, 0.3, 3.14), SE2Pose(1.6, 0.3, 1.57)]` in a sequence, which is the absolute pose defined in the world.
