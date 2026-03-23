import mujoco
import mujoco.viewer
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
import numpy as np

# can be controlled over rclpy with `ros2 topic pub /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.0, 1.0]}"` (tendon mode)
# or `ros2 topic pub /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.1, 0.2, ...]}"` (joint mode)

# ros2 topic pub --once /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0]}"

class TentacleSimNode(Node):
    def __init__(self):
        super().__init__('tentacle_mujoco_sim')
        
        # Create a subscriber to listen for tendon or joint commands
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/tentacle/tendon_cmds',
            self.cmd_callback,
            10)
        
        # Store the latest commands
        self.latest_cmds = []
        
        # Publisher for RViz visualization and feedback
        self.marker_pub = self.create_publisher(MarkerArray, '/tentacle/tendon_markers', 10)
        self.tip_pose_pub = self.create_publisher(PoseStamped, '/tentacle/tip_pose', 10)
        self.tip_path_pub = self.create_publisher(Path, '/tentacle/tip_path', 10)
        self.tendon_len_pub = self.create_publisher(Float64MultiArray, '/tentacle/tendon_lengths', 10)
        
        self.tip_path_msg = Path()
        
        # Boolean toggle for RViz publishing
        self.publish_rviz = True
        
        self.get_logger().info("Tentacle Sim Node Started. Listening on /tentacle/tendon_cmds")

    def cmd_callback(self, msg):
        # Update our internal state when a new ROS message arrives
        self.latest_cmds = list(msg.data)

    def publish_rviz_markers(self, model, data):
        """Publishes tendon markers, tip pose, and tip path to RViz."""
        if not self.publish_rviz:
            return

        # Publish RViz Markers (at ~30 Hz)
        if int(data.time * 30) % 1 == 0 and data.time % (1.0/30.0) < model.opt.timestep:
            now_msg = self.get_clock().now().to_msg()
            
            # 1. Publish Tendon Markers
            array_msg = MarkerArray()
            for ten_id in range(model.ntendon):
                rgba = model.tendon_rgba[ten_id]
                
                start = data.ten_wrapadr[ten_id]
                num = data.ten_wrapnum[ten_id]
                
                current_marker = Marker()
                current_marker.header.frame_id = "world"
                current_marker.header.stamp = now_msg
                current_marker.ns = f"tendon_{ten_id}"
                current_marker.id = 0
                current_marker.type = Marker.LINE_STRIP
                current_marker.action = Marker.ADD
                current_marker.color.r = float(rgba[0])
                current_marker.color.g = float(rgba[1])
                current_marker.color.b = float(rgba[2])
                current_marker.color.a = float(rgba[3])
                current_marker.scale.x = 0.01
                
                sub_id = 0
                prev_p = None
                for i in range(start, start + num):
                    p = Point()
                    p.x = float(data.wrap_xpos[i, 0])
                    p.y = float(data.wrap_xpos[i, 1])
                    p.z = float(data.wrap_xpos[i, 2])
                    
                    if prev_p is not None:
                        dist = ((p.x - prev_p.x)**2 + (p.y - prev_p.y)**2 + (p.z - prev_p.z)**2)**0.5
                        if dist > 0.05: # Large jump threshold (5cm)
                            array_msg.markers.append(current_marker)
                            sub_id += 1
                            current_marker = Marker()
                            current_marker.header.frame_id = "world"
                            current_marker.header.stamp = now_msg
                            current_marker.ns = f"tendon_{ten_id}"
                            current_marker.id = sub_id
                            current_marker.type = Marker.LINE_STRIP
                            current_marker.action = Marker.ADD
                            current_marker.color.r = float(rgba[0])
                            current_marker.color.g = float(rgba[1])
                            current_marker.color.b = float(rgba[2])
                            current_marker.color.a = float(rgba[3])
                            current_marker.scale.x = 0.01
                            
                    current_marker.points.append(p)
                    prev_p = p
                
                if len(current_marker.points) > 0:
                    array_msg.markers.append(current_marker)
            
            self.marker_pub.publish(array_msg)
            
            # 2. Publish Tip Pose and Path
            tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link24")
            if tip_id != -1:
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = now_msg
                
                pose_msg.pose.position.x = float(data.xpos[tip_id, 0])
                pose_msg.pose.position.y = float(data.xpos[tip_id, 1])
                pose_msg.pose.position.z = float(data.xpos[tip_id, 2])
                
                pose_msg.pose.orientation.w = float(data.xquat[tip_id, 0])
                pose_msg.pose.orientation.x = float(data.xquat[tip_id, 1])
                pose_msg.pose.orientation.y = float(data.xquat[tip_id, 2])
                pose_msg.pose.orientation.z = float(data.xquat[tip_id, 3])
                
                self.tip_pose_pub.publish(pose_msg)
                
                self.tip_path_msg.header = pose_msg.header
                self.tip_path_msg.poses.append(pose_msg)
                
                if len(self.tip_path_msg.poses) > 500:
                    self.tip_path_msg.poses.pop(0)
                    
                self.tip_path_pub.publish(self.tip_path_msg)

def main():
    # 1. Initialize ROS 2
    rclpy.init()
    ros_node = TentacleSimNode()

    # 2. Load the MuJoCo model
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "tentacle.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 3. Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation started. Press Space to pause/unpause.")
        print(f"Model has {model.nu} actuators.")
        print(f"  First 2 are Tendon Muscles (0 to 1)")
        print(f"  Next 23 are Joint Position Controllers (-1 to 1 maps to -30 to 30 deg)")
        
        while viewer.is_running() and rclpy.ok():
            step_start = time.time()

            # --- A. Process ROS 2 Callbacks ---
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # --- B. Apply ROS Commands to MuJoCo ---
            if len(ros_node.latest_cmds) == 2:
                # Tendon mode (act1, act2)
                data.ctrl[0] = ros_node.latest_cmds[0]
                data.ctrl[1] = ros_node.latest_cmds[1]
                # Zero out joint actuators
                data.ctrl[2:] = 0
            elif len(ros_node.latest_cmds) >= 23:
                # Joint mode (adj1, adj2, ..., adj23)
                # Zero out tendon actuators
                data.ctrl[0:2] = 0
                # Scale -1..1 to -30..30 degrees (0.523599 radians)
                cmds = np.array(ros_node.latest_cmds[:23])
                scaled_cmds = cmds * 0.523599
                data.ctrl[2:25] = scaled_cmds

            # --- C. Step the Physics ---
            mujoco.mj_step(model, data)

            # --- D. Print and Publish Tendon Lengths ---
            if int(data.time * 10) % 1 == 0 and data.time % 0.1 < model.opt.timestep:
                # Tendon lengths (from act1, act2)
                l1 = float(data.actuator_length[0])
                l2 = float(data.actuator_length[1])
                
                # Publish
                len_msg = Float64MultiArray()
                len_msg.data = [l1, l2]
                ros_node.tendon_len_pub.publish(len_msg)

                if len(ros_node.latest_cmds) >= 23:
                    mode_str = "Joint"
                else:
                    mode_str = "Tendon"
                print(f"Time: {data.time:.2f} | Mode: {mode_str} | L1={l1:.4f}, L2={l2:.4f}")

            # --- E. Publish RViz Markers (at ~30 Hz) ---
            ros_node.publish_rviz_markers(model, data)

            # --- F. Sync Visuals and Real-time Clock ---
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Cleanup when the window is closed
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
