#! /usr/bin/env python3

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from .model import load_dronet
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import math

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacleavoidance')
        self.sub_image_raw = self.create_subscription(Image, '/zed_camera/left/image_raw', self.image_callback, 10)
        model_path = '/home/ava/PycharmProjects/Obstacle-Avoidance-in-ROS2/obstacle_avoidance/models/dronet/model_struct.json'
        weights_path = '/home/ava/PycharmProjects/Obstacle-Avoidance-in-ROS2/obstacle_avoidance/models/dronet/best_weights.h5'
        self.pub_resized_image = self.create_publisher(Image, '/image/resized_image', 1)
        self.dronet = load_dronet(model_path, weights_path)
        self.frame_sequence = 1
        self.velocity_publisher = VelocityPublisher()
        self.bridge = CvBridge()

        # define forward velocity coefficients
        self.alpha = 0.5
        self.beta = 0.7
        # define backward velocity coefficients
        self.gamma = 0.5

        # define thresholds of hystersis thresholding
        self.threshold1 = 0.95
        self.threshold2 = 0.99

        # initialize variables
        self.s_prev = 0
        self.v_prev = 0
        self.s_current = 0
        self.v_current = 0
        self.go_backward = False
        self.s_prev_backward = 0

        # define constants
        self.v_max = 0.5
        self.s_max = math.pi / 6
        self.v_back = -0.5
        self.s_back = -0.3
        self.cropped_rows = 300
        self.velocity_publisher.publish(0.5, self.v_current)

    # After processing it publishes back the estimated depth result
    def image_callback(self, msg):
        # Convert message to opencv image
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        resized_img = cv2.resize(image[:self.cropped_rows+1,:], (200, 200))
        # resized_img = cv2.resize(image, (200, 200))
        resized_msg = self.bridge.cv2_to_imgmsg(resized_img, encoding="passthrough")
        self.pub_resized_image.publish(resized_msg)

        gray_frame = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        input_frame = gray_frame / 255.0
        prediction = self.dronet.predict(np.expand_dims(input_frame, axis=0))
        # Visualize the result
        steering_angle, collision_probability = prediction[0][0][0], prediction[1][0][0]
    
        self.compute_velocity(steering_angle, collision_probability)
        self.velocity_publisher.publish(self.v_current, self.s_current)
        # self.get_logger().info('collision_prob: "%.2f" steering_mean: "%.2f" velocity: "%.2f"' % (collision_probability, self.s_current, self.v_current))
        self.get_logger().info('collision: "%.2f" angle: "%.2f" velocity: "%.2f"' % (collision_probability, self.s_current, self.v_current))

    def compute_velocity(self, steering_angle, collision_prob):

        self.s_prev = self.s_current
        self.v_prev = self.v_current
        
        if collision_prob <= self.threshold1:
            self.compute_forward_velocity(steering_angle, collision_prob)
            self.go_backward = False
            self.s_prev_backward = self.s_current
        elif collision_prob >= self.threshold2:
            self.compute_backward_velocity()
            self.go_backward = True
            
        else:
            if self.go_backward:
                self.compute_backward_velocity()
            else:
                self.compute_forward_velocity(steering_angle, collision_prob)
    
    def compute_forward_velocity(self, steering_angle, collision_prob):
        self.v_current = ((1-self.alpha) * self.v_prev) + (self.alpha * (1-collision_prob) * self.v_max)
        self.s_current = ((1-self.beta) * self.s_prev) + (self.beta * self.s_max * steering_angle)
    
    def compute_backward_velocity(self):
        self.s_prev_back = self.s_current
        self.v_current = self.v_back
        # self.s_current = (1-self.gamma) * self.s_prev_back + self.gamma * self.s_back
        self.s_current = self.s_back


class VelocityPublisher(Node):

    def __init__(self):
        super().__init__('publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

    def publish(self, linear_x_velocity, angular_z_velocity):
        msg = Twist()
        msg.linear.x = float(linear_x_velocity)
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z_velocity)


        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    depth_node = ObstacleAvoidance()
    rclpy.spin(depth_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
