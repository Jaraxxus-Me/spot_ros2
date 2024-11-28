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
WORLD_GOAL = [SE2Pose(1.8, 0.3, 3.14), SE2Pose(1.6, 0.3, 1.57)]

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
                            relative_pose: math_helpers.SE2Pose,
                            max_xytheta_vel = (2.0, 2.0, 1.0),
                            min_xytheta_vel = (-2.0, -2.0, 1.0)) -> None:
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
