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
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, WR1_FRAME_NAME, \
    get_se2_a_tform_b
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

# Where we want the robot to walk to relative to itself
WORLD_GOAL = [SE2Pose(1.8, 0.4, 3.14), SE2Pose(1.6, 0.4, 1.23)]

NUM_LOCALIZATION_RETRIES = 10
LOCALIZATION_RETRY_WAIT_TIME = 1.0


class LocalizationFailure(Exception):
    """Raised when localization fails."""

class SpotLocalizer:
    """Localizes spot in a previously mapped environment."""

    def __init__(self, robot: Robot, upload_path: str) -> None:
        self._robot = robot
        self._upload_path = upload_path

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create the client for the Graph Nav main service.
        self.graph_nav_client = self._robot.ensure_client(
            GraphNavClient.default_service_name)

        # Upload graph and snapshots on start.
        self._upload_graph_and_snapshots()

        # Initialize robot pose, which will be updated in localize().
        self._robot_pose = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat())
        # Initialize the robot's position in the map.
        robot_state = get_robot_state(self._robot)
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        localization = nav_pb2.Localization()
        for r in range(NUM_LOCALIZATION_RETRIES + 1):
            try:
                self.graph_nav_client.set_localization(
                    initial_guess_localization=localization,
                    ko_tform_body=current_odom_tform_body)
                break
            except (ResponseError, TimedOutError) as e:
                # Retry or fail.
                if r == NUM_LOCALIZATION_RETRIES:
                    msg = f"Localization failed permanently: {e}."
                    logging.warning(msg)
                    raise LocalizationFailure(msg)
                logging.warning("Localization failed once, retrying.")
                time.sleep(LOCALIZATION_RETRY_WAIT_TIME)

        # Run localize once to start.
        self.localize()

    def _upload_graph_and_snapshots(self) -> None:
        """Upload the graph and snapshots to the robot."""
        # pylint: disable=no-member
        logging.info("Loading the graph from disk into local storage...")
        # Load the graph from disk.
        with open(self._upload_path / "graph", "rb") as f:
            data = f.read()
            current_graph = map_pb2.Graph()
            current_graph.ParseFromString(data)
            logging.info(f"Loaded graph has {len(current_graph.waypoints)} "
                         f"waypoints and {len(current_graph.edges)} edges")
        # Load the waypoint snapshots from disk.
        waypoint_path = self._upload_path / "waypoint_snapshots"
        waypoint_snapshots: Dict[str, map_pb2.WaypointSnapshot] = {}
        for waypoint in current_graph.waypoints:
            with open(waypoint_path / waypoint.snapshot_id, "rb") as f:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(f.read())
                waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        # Load the edge snapshots from disk.
        edge_path = self._upload_path / "edge_snapshots"
        edge_snapshots: Dict[str, map_pb2.EdgeSnapshot] = {}
        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            with open(edge_path / edge.snapshot_id, "rb") as f:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(f.read())
                edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        logging.info("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(current_graph.anchoring.anchors)
        response = self.graph_nav_client.upload_graph(
            graph=current_graph, generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = waypoint_snapshots[snapshot_id]
            self.graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = edge_snapshots[snapshot_id]
            self.graph_nav_client.upload_edge_snapshot(edge_snapshot)

    def get_last_robot_pose(self) -> math_helpers.SE3Pose:
        """Get the last estimated robot pose.

        Does not localize.
        """
        return self._robot_pose

    def localize(self,
                 num_retries: int = 10,
                 retry_wait_time: float = 1.0) -> None:
        """Re-localize the robot and return the current SE3Pose of the body.

        It's good practice to call this periodically to avoid drift
        issues. April tags need to be in view.
        """
        try:
            localization_state = self.graph_nav_client.get_localization_state()
            transform = localization_state.localization.seed_tform_body
            if str(transform) == "":
                raise LocalizationFailure("Received empty localization state.")
        except (ResponseError, TimedOutError, LocalizationFailure) as e:
            # Retry or fail.
            if num_retries <= 0:
                msg = f"Localization failed permanently: {e}."
                logging.warning(msg)
                raise LocalizationFailure(msg)
            logging.warning("Localization failed once, retrying.")
            time.sleep(retry_wait_time)
            return self.localize(num_retries=num_retries - 1,
                                 retry_wait_time=retry_wait_time)
        logging.info("Localization succeeded.")
        self._robot_pose = math_helpers.SE3Pose.from_proto(transform)
        return None

class Localize:
    def __init__(self, hostname: str, \
                 robot_name: Optional[str] = None, node: Optional[Node] = None) -> None:
        self._logger = logging.getLogger(fqn(self.__class__))
        node = node or ros_scope.node()
        self.finished = False
        if node is None:
            raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")
        self._robot_name = robot_name
        self._robot_ros = SimpleSpotCommander(self._robot_name, node)
        self.body_frame_name = namespace_with(robot_name, BODY_FRAME_NAME)
        self.hand_frame_name = namespace_with(robot_name, HAND_FRAME_NAME)
        self.tf_listener = TFListenerWrapper(node)
        self.tf_listener.wait_for_a_tform_b(self.body_frame_name, self.hand_frame_name)
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
        

    def initialize_robot(self) -> bool:
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
        # calculate the hand pose in the world frame
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

        world_T_body = self.localizer.get_last_robot_pose()
        world_T_hand = world_T_body * body_T_hand_se3
        self._logger.info(f"Hand pose in world frame: {world_T_hand}")
        return True


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="192.168.80.3")
    parser.add_argument("--robot", type=str, default=None)
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    localizer = Localize(args.hostname, args.robot, main.node)
    localizer.initialize_robot()
    return 0


if __name__ == "__main__":
    exit(main())
