import os
import argparse
import logging
import time
import numpy as np
from typing import Optional

# use ROS 2 Python API
import bdai_ros2_wrappers.process as ros_process
import bdai_ros2_wrappers.scope as ros_scope
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.utilities import fqn, namespace_with
# use Boston Dynamics Python API
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.graph_nav import map_pb2, nav_pb2
from bosdyn.client import ResponseError, TimedOutError, math_helpers
from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate
from bosdyn_msgs.conversions import convert
from rclpy.node import Node
# use IsaacPlan helpers
from IsaacPlan import utils
from IsaacPlan.spot_utils.utils import get_robot_state
from IsaacPlan.spot_utils.utils import get_graph_nav_dir, verify_estop

from spot_msgs.action import RobotCommand  # type: ignore

from .simple_spot_commander import SimpleSpotCommander
from .localize import SpotLocalizer

# Where we want the robot to walk to relative to itself
# WORLD_GOAL = [SE2Pose(1.8, 0.4, 3.14), SE2Pose(1.6, 0.4, 1.23)]
WORLD_GOAL = [SE2Pose(2.2, 0.6, 3.141)]

NUM_LOCALIZATION_RETRIES = 10
LOCALIZATION_RETRY_WAIT_TIME = 1.0

class WalkAround:
    def __init__(self, hostname: str, \
                 robot_name: Optional[str] = None, node: Optional[Node] = None) -> None:
        self._logger = logging.getLogger(fqn(self.__class__))
        node = node or ros_scope.node()
        self.finished = False
        if node is None:
            raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")
        self._robot_name = robot_name
        self._robot_ros = SimpleSpotCommander(self._robot_name, node)
        self._robot_command_client = ActionClientWrapper(
            RobotCommand, namespace_with(self._robot_name, "robot_command"), node
        )
        self._body_frame_name = namespace_with(self._robot_name, BODY_FRAME_NAME)
        self._odom_frame_name = namespace_with(self._robot_name, ODOM_FRAME_NAME)
        self._tf_listener = TFListenerWrapper(node)
        self._tf_listener.wait_for_a_tform_b(self._body_frame_name, self._odom_frame_name)
        # Create the SDK and robot objects.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        utils.reset_config({
        "spot_graph_nav_map": "debug_place"
        })
        sdk = create_standard_sdk('GraphNavTestClient')
        path = get_graph_nav_dir()
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        assert os.path.exists(path), f"Graph path {path} does not exist"
        self._logger.info(f"Using graph path {path}")
        self.localizer = SpotLocalizer(robot, path)
        self._robot_sdk = robot
        # Store the goal points in the world frame.
        self._world_goals = [goal for goal in WORLD_GOAL]
        

    def initialize_robot(self) -> bool:
        self._logger.info(f"Robot name: {self._robot_name}")
        self._logger.info("Claiming robot")
        result = self._robot_ros.command("claim")
        if not result.success:
            self._logger.error("Unable to claim robot message was " + result.message)
            return False
        self._logger.info("Claimed robot")

        # Stand the robot up.
        self._logger.info("Powering robot on")
        result = self._robot_ros.command("power_on")
        if not result.success:
            self._logger.error("Unable to power on robot message was " + result.message)
            return False
        self._logger.info("Standing robot up")
        result = self._robot_ros.command("stand")
        if not result.success:
            self._logger.error("Robot did not stand message was " + result.message)
            return False
        self._logger.info("Successfully stood up.")
        # Localize the robot.
        world_pose = self.localizer.get_last_robot_pose()
        self._logger.info(f"Robot localized at {world_pose}")
        return True

    def walk_to_nex_goal(self) -> None:
        self._logger.info("Walking To Goal {}".format(len(WORLD_GOAL) - \
                                    len(self._world_goals)))
        self._logger.info("{}".format(self._world_goals[0]))
        self.localizer.localize()
        current_se2_pose = self.localizer.get_last_robot_pose().get_closest_se2_transform()
        self._logger.info(f"Robot pose in world frame: {current_se2_pose}")
        rel_pose = current_se2_pose.inverse() * self._world_goals[0]
        self.navigate_to_relative_pose(rel_pose)
        # check current pose
        self.localizer.localize()
        current_se2_pose = self.localizer.get_last_robot_pose().get_closest_se2_transform()
        dis = np.array([current_se2_pose.x - self._world_goals[0].x,
                        current_se2_pose.y - self._world_goals[0].y])
        dis = np.linalg.norm(dis)
        angle_error = abs(current_se2_pose.angle - self._world_goals[0].angle)
        if dis < 0.1 and angle_error < 0.1:
            self._logger.info("Successfully walked to next goal")
            self._world_goals.pop(0)
            if len(self._world_goals) == 0:
                self.finished = True

    def navigate_to_relative_pose(self,
                            relative_pose: math_helpers.SE2Pose) -> None:
        """Execute a relative move.

        The pose is dx, dy, dyaw relative to the robot's body.
        """
        # Get the robot's current state.
        robot_state = get_robot_state(self._robot_sdk)
        transforms = robot_state.kinematic_state.transforms_snapshot
        assert str(transforms) != ""

        # We do not want to command this goal in body frame because the body will
        # move, thus shifting our goal. Instead, we transform this offset to get
        # the goal position in the output frame (odometry).
        odom_t_body = self._tf_listener.lookup_a_tform_b(self._odom_frame_name, self._body_frame_name)
        odom_t_robot_se2 = SE3Pose(
            odom_t_body.transform.translation.x,
            odom_t_body.transform.translation.y,
            odom_t_body.transform.translation.z,
            Quat(
                odom_t_body.transform.rotation.w,
                odom_t_body.transform.rotation.x,
                odom_t_body.transform.rotation.y,
                odom_t_body.transform.rotation.z,
            ),
        ).get_closest_se2_transform()
        self._logger.info(f"Relative pose: {relative_pose}")
        out_tform_goal = odom_t_robot_se2 * relative_pose

        # Command the robot to go to the goal point in the specified
        # frame. The command will stop at the new position.
        # Constrain the robot not to turn, forcing it to strafe laterally.
        # speed_limit = SE2VelocityLimit(
        #     max_vel=SE2Velocity(linear=Vec2(x=max_xytheta_vel[0],
        #                                     y=max_xytheta_vel[1]),
        #                         angular=max_xytheta_vel[2]),
        #     min_vel=SE2Velocity(linear=Vec2(x=min_xytheta_vel[0],
        #                                     y=min_xytheta_vel[1]),
        #                         angular=min_xytheta_vel[2]))
        # mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)
        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME)
        logging.info(f"Sending goal: {out_tform_goal.x}, {out_tform_goal.y}, {out_tform_goal.angle}")
        action_goal = RobotCommand.Goal()
        convert(proto_goal, action_goal.command)
        self._robot_command_client.send_goal_and_wait("walk_forward", action_goal)

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="192.168.80.3")
    parser.add_argument("--robot", type=str, default=None)
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    walker = WalkAround(args.hostname, args.robot, main.node)
    walker.initialize_robot()
    while not walker.finished:
        walker.walk_to_nex_goal()
    return 0


if __name__ == "__main__":
    exit(main())
