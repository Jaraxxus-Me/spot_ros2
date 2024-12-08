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
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.graph_nav import map_pb2, nav_pb2
from bosdyn.client import ResponseError, TimedOutError, math_helpers
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate
from bosdyn.client.sdk import Robot
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn_msgs.conversions import convert
from rclpy.node import Node
# use IsaacPlan helpers
from IsaacPlan import utils
from IsaacPlan.spot_utils.utils import get_robot_state
from IsaacPlan.spot_utils.utils import get_graph_nav_dir, verify_estop

from spot_msgs.action import RobotCommand  # type: ignore

from .simple_spot_commander import SimpleSpotCommander
from .walk_around_localize import SpotLocalizer

# Where we want the robot to walk to relative to itself
HYBRID_DELTA = [
    {
        "body": SE3Pose(0.3, -0.1, 0, Quat(0, 0, 0, 1)),
        "hand": SE2Pose(0.3, 0.0, 0, Quat(0, 0, 0, 1))
    },
    {
        "body": SE2Pose(-0.2, -0.2, 0, Quat.from_yaw(30)),
        "hand": SE2Pose(0.1, 0.1, 0, Quat.from_pitch(30))
    }
]


class WalkManiuplate:
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
        self.body_frame_name = namespace_with(robot_name, BODY_FRAME_NAME)
        self.hand_frame_name = namespace_with(robot_name, HAND_FRAME_NAME)
        self.odom_frame_name = namespace_with(robot_name, ODOM_FRAME_NAME)
        self._tf_listener = TFListenerWrapper(node)
        self._tf_listener.wait_for_a_tform_b(self.body_frame_name, self.hand_frame_name)
        self._tf_listener.wait_for_a_tform_b(self.body_frame_name, self.odom_frame_name)
        self._tf_listener.wait_for_a_tform_b(self.hand_frame_name, self.odom_frame_name)
        # Create the SDK and robot objects.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        utils.reset_config({
        "spot_graph_nav_map": "desk_debug"
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
        self.accomplished_world_goals = [False, False] * HYBRID_DELTA
        self.at = 0
        

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
        time.wait(1)
        _, _ = self.localize_body_hand()
        return True
    
    def localize_body_hand(self) -> None:
        body_T_hand = self.tf_listener.lookup_a_tform_b(self.body_frame_name, self.hand_frame_name)
        body_T_hand_se3 = math_helpers.SE3Pose(
            body_T_hand.transform.translation.x,
            body_T_hand.transform.translation.y,
            body_T_hand.transform.translation.z,
            math_helpers.Quat(
                body_T_hand.transform.rotation.w,
                body_T_hand.transform.rotation.x,
                body_T_hand.transform.rotation.y,
                body_T_hand.transform.rotation.z,
            ),
        )
        self.localizer.localize()
        world_T_body = self.localizer.get_last_robot_pose()
        self._logger.info(f"Robot localized at {world_T_body}")
        world_T_hand = world_T_body * body_T_hand_se3
        self._logger.info(f"Hand pose in world frame: {world_T_hand}")
        return world_T_body, world_T_hand

    def achieve_next_goal(self) -> None:
        self._logger.info("Achieving Goal {}".format(len(self.at)))
        body_finished = False
        hand_finished = False
        self._logger.info("{}".format(HYBRID_DELTA[self.at]))
        self.localizer.localize()
        current_se3_pose = self.localizer.get_last_robot_pose()
        self._logger.info(f"Robot pose in world frame: {current_se3_pose}")
        rel_pose = HYBRID_DELTA[self.at]["body"]
        desired_pose = (current_se3_pose @ rel_pose).get_closest_se2_transform()
        self.navigate_to_relative_pose(rel_pose)
        # check current pose
        self.localizer.localize()
        current_se2_pose = self.localizer.get_last_robot_pose().get_closest_se2_transform()
        dis = np.array([current_se2_pose.x - desired_pose.x,
                        current_se2_pose.y - desired_pose.y])
        dis = np.linalg.norm(dis)
        angle_error = abs(current_se2_pose.angle - desired_pose.angle)
        if dis < 0.1 and angle_error < 0.1:
            self._logger.info("Successfully walked to next goal")
            body_finished = True
            _, hand_pose = self.localize_body_hand()
            rel_pose = HYBRID_DELTA[self.at]["hand"]
            desired_pose = hand_pose @ rel_pose
            self.move_hand_to_relative_pose(rel_pose)
            # check current pose
            _, hand_pose = self.localize_body_hand()
            dis = np.array([hand_pose.x - desired_pose.x,
                            hand_pose.y - desired_pose.y,
                            hand_pose.z - desired_pose.z])
            dis = np.linalg.norm(dis)
            if dis < 0.1:
                self._logger.info("Successfully moved hand to next goal")
                hand_finished = True
        if body_finished and hand_finished:
            self.accomplished_world_goals[self.at] = [True, True]
            self.at += 1

    def move_hand_to_relative_pose(self,
            relative_pose: math_helpers.SE3Pose) -> None:
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
        odom_t_hand = self._tf_listener.lookup_a_tform_b(self.odom_frame_name, self.hand_frame_name)
        odom_T_hand = SE3Pose(
            odom_t_hand.transform.translation.x,
            odom_t_hand.transform.translation.y,
            odom_t_hand.transform.translation.z,
            Quat(
                odom_t_hand.transform.rotation.w,
                odom_t_hand.transform.rotation.x,
                odom_t_hand.transform.rotation.y,
                odom_t_hand.transform.rotation.z,
            ),
        )

        odom_T_new_hand = odom_T_hand @ relative_pose
        self._logger.info(f"Relative pose: {relative_pose}")
        out_tform_goal = odom_t_robot_se2 * relative_pose
        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME)
        logging.info(f"Sending goal: {out_tform_goal.x}, {out_tform_goal.y}, {out_tform_goal.angle}")
        action_goal = RobotCommand.Goal()
        convert(proto_goal, action_goal.command)
        self._robot_command_client.send_goal_and_wait("walk_forward", action_goal)

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
        odom_t_body = self._tf_listener.lookup_a_tform_b(self.odom_frame_name, self.body_frame_name)
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
        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME)
        logging.info(f"Sending goal: {out_tform_goal.x}, {out_tform_goal.y}, {out_tform_goal.angle}")
        action_goal = RobotCommand.Goal()
        convert(proto_goal, action_goal.command)
        self._robot_command_client.send_goal_and_wait("walk_forward", action_goal)

    def finished(self) -> bool:
        for goal in self.accomplished_world_goals:
            for subgoal in goal:
                if not subgoal:
                    return False
        return True

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="192.168.80.3")
    parser.add_argument("--robot", type=str, default=None)
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    walker = WalkManiuplate(args.hostname, args.robot, main.node)
    walker.initialize_robot()
    while not walker.finished():
        time.wait(1)
        walker.achieve_next_goal()
    return 0


if __name__ == "__main__":
    exit(main())
