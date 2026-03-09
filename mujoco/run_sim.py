import mujoco
import mujoco.viewer
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# can be controlled over rclpy with `ros2 topic pub /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.0, 1.0]}"`

class TentacleSimNode(Node):
    def __init__(self):
        super().__init__('tentacle_mujoco_sim')
        
        # Create a subscriber to listen for tendon commands
        # We use a Float64MultiArray to send [act1_value, act2_value]
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/tentacle/tendon_cmds',
            self.cmd_callback,
            10)
        
        # Store the latest commands (default to 0 tension)
        self.latest_cmds = [0.0, 0.0]
        self.get_logger().info("Tentacle Sim Node Started. Listening on /tentacle/tendon_cmds")

    def cmd_callback(self, msg):
        # Update our internal state when a new ROS message arrives
        if len(msg.data) >= 2:
            self.latest_cmds = [msg.data[0], msg.data[1]]

def main():
    # 1. Initialize ROS 2
    rclpy.init()
    ros_node = TentacleSimNode()

    # 2. Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("tentacle.xml")
    data = mujoco.MjData(model)

    # 3. Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation started. Press Space to pause/unpause.")
        
        while viewer.is_running() and rclpy.ok():
            step_start = time.time()

            # --- A. Process ROS 2 Callbacks ---
            # timeout_sec=0.0 makes this non-blocking. It just checks for messages and moves on.
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # --- B. Apply ROS Commands to MuJoCo ---
            # Apply the commands received from the ROS topic to the MuJoCo actuators
            data.ctrl[0] = ros_node.latest_cmds[0]
            data.ctrl[1] = ros_node.latest_cmds[1]

            # --- C. Step the Physics ---
            mujoco.mj_step(model, data)

            # --- D. Print Debug Info ---
            if int(data.time * 10) % 1 == 0 and data.time % 0.1 < model.opt.timestep:
                # Print the length of the strings
                print(f"Time: {data.time:.2f} | Tendon Lengths: L={data.actuator_length[0]:.4f}, R={data.actuator_length[1]:.4f}")

            # --- E. Sync Visuals and Real-time Clock ---
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Cleanup when the window is closed
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()