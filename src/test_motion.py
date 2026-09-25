import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time
import math

class TentacleController(Node):
    def __init__(self):
        super().__init__('tentacle_motion_tester')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/tentacle/tendon_cmds', 10)
        self.get_logger().info("Tentacle Motion Tester Started (Traveling Wave).")

    def publish_cmds(self, data):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in data]
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = TentacleController()

    num_joints = 23
    
    print("\n--- Tentacle Motion: Traveling Wave ---")
    print("Running crawling pattern. Press Ctrl+C to stop.")
    
    start_time = time.time()
    try:
        while rclpy.ok():
            t = time.time() - start_time
            cmds = [0.0] * num_joints
            
            # Traveling Wave pattern (worked best for the user)
            for i in range(num_joints):
                # Frequency: 0.5 Hz, Phase Shift: 0.4 rad per link
                cmds[i] = math.sin(math.pi * t - 0.4 * i)
            
            node.publish_cmds(cmds)
            time.sleep(0.02)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
