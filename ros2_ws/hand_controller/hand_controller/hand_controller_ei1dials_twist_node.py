# Copyright 2025 Tria Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys
import os
import importlib

from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

#from std_msgs.msg import String
from geometry_msgs.msg import Twist

#
# Tria color palette
#

# Primary Palette (RGB format)
tria_blue   = (  0,  31,  99); # TRIA BLUE
tria_pink   = (255,   0, 163); # TRIA PINK
tria_white  = (255, 255, 255); # WHITE

# Secondary Palette (RGB format)
tria_gray11 = ( 83,  86,  90); # COOL GRAY 11
tria_gray7  = (151, 153, 155); # COOL GRAY 7
tria_gray3  = (200, 201, 199); # COOL GRAY 3

# Tertiary Palette (RGB format)
tria_purple = (107,  83, 157); # TRIA PURPLE
tria_yellow = (235, 201,  80); # TRIA YELLOW
tria_aqua   = (  0, 161, 190); # TRIA AQUA
tria_black  = (  0,   0,   0); # BLACK

# Edge Impulse

from edge_impulse_linux.image import ImageImpulseRunner


def resize_pad(img,h_scale,w_scale):
    """ resize and pad images to be input to the detectors

    The FOMO model take 160x160 images as input. 
    As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img: HxW
        scale: scale factor between original image and 160x160 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0] >= size0[1]:
        h1 = int(h_scale)
        w1 = int(w_scale * size0[1] // size0[0])
        padh = 0
        padw = int(w_scale - w1)
        scale = size0[1] / w1
    else:
        h1 = int(h_scale * size0[0] // size0[1])
        w1 = int(w_scale)
        padh = int(h_scale - h1)
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh % 2
    padw1 = padw//2
    padw2 = padw//2 + padw % 2
    img = cv2.resize(img, (w1, h1))
    img = np.pad(img, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
    pad = (int(padh1 * scale), int(padw1 * scale))
    return img, scale, pad


# Visual Control Dials

#CV_DRAW_COLOR_PRIMARY = (255, 255, 0)
CV_DRAW_COLOR_PRIMARY = tria_aqua

CONTROL_CIRCLE_DEADZONE_R = 50

@dataclass
class HandData:
    handedness: str
    landmarks: list
    center_perc: tuple

    def __init__(self, handedness, landmarks, image_width, image_height):
        self.handedness = handedness
        self.landmarks = landmarks.copy()
        self.landmarks[:,0] = self.landmarks[:,0] / image_width
        self.landmarks[:,1] = self.landmarks[:,1] / image_height        
        landmarks_len = landmarks.shape[0]
        x_avg = sum(self.landmarks[:,0]) / landmarks_len
        y_avg = sum(self.landmarks[:,1]) / landmarks_len
        z_avg = sum(self.landmarks[:,2]) / landmarks_len
        
        self.center_perc = (x_avg, y_avg, z_avg)
        #print(f"center_perc: {self.center_perc}")

def draw_control_overlay(img, lh_data=None, rh_data=None):
    CAMERA_HEIGHT, CAMERA_WIDTH, _ = img.shape

    CONTROL_CIRCLE_XY_CENTER = (int(CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))
    CONTROL_CIRCLE_Z_APERATURE_CENTER = (int(3*CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))
    
    # Draw control circle for XY control (left hand)
    cv2.circle(img, CONTROL_CIRCLE_XY_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    center_xy_point = CONTROL_CIRCLE_XY_CENTER
    hand_xy_point = CONTROL_CIRCLE_XY_CENTER # until proven otherwise

    if lh_data:
        # Normalize and compute actual pixel position of left hand
        xy_ctl_x_pct_normalized = min((lh_data.center_perc[0] - 0.25) * 4, 1.0)
        xy_ctl_y_pct_normalized = min((lh_data.center_perc[1] - 0.5) * 2, 1.0)

        xy_ctl_x = int(xy_ctl_x_pct_normalized *
                       CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[0]
        xy_ctl_y = int(xy_ctl_y_pct_normalized *
                       CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[1]

        hand_xy_point = (xy_ctl_x, xy_ctl_y)

        # Draw line from center to hand position
        cv2.line(img, center_xy_point, hand_xy_point,
                 CV_DRAW_COLOR_PRIMARY, 1)

        # Draw hand position dot
        cv2.circle(img, hand_xy_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    # Calculate normalized delta values
    delta_xy = tuple(np.subtract(center_xy_point,hand_xy_point))
    delta_xy = tuple(c/CONTROL_CIRCLE_DEADZONE_R for c in delta_xy)

    # Draw control circle for Z-aperture (right hand)
    cv2.circle(img, CONTROL_CIRCLE_Z_APERATURE_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    center_z_point = CONTROL_CIRCLE_Z_APERATURE_CENTER
    hand_z_point = CONTROL_CIRCLE_Z_APERATURE_CENTER # until proven otherwise

    if rh_data:
        z_ctl_pct_normalized = min((rh_data.center_perc[1] - 0.50) * 2, 1.0)
        aperature_ctl_x_pct_normalized = min(
            (rh_data.center_perc[0] - 0.75) * 4, 1.0)

        aperature_ctl_x = int(aperature_ctl_x_pct_normalized *
                              CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[0]
        z_ctl_y = int(z_ctl_pct_normalized *
                      CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[1]

        hand_z_point = (aperature_ctl_x, z_ctl_y)

        # Draw line from center to hand Z-position
        cv2.line(img, center_z_point, hand_z_point,
                 CV_DRAW_COLOR_PRIMARY, 1)

        # Draw hand position dot
        cv2.circle(img, hand_z_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    # Calculate normalized delta values
    delta_z = tuple(np.subtract(center_z_point,hand_z_point))
    delta_z = tuple(c/CONTROL_CIRCLE_DEADZONE_R for c in delta_z)

    # Optional: draw vertical center reference line
    cv2.line(img, (int(CAMERA_WIDTH / 2), 0),
             (int(CAMERA_WIDTH / 2), CAMERA_HEIGHT), CV_DRAW_COLOR_PRIMARY, 1)

    return delta_xy, delta_z

class HandControllerEi1DialsTwistNode(Node):

    def __init__(self):
        super().__init__('hand_controller_ei1dials_twist_node')
        self.subscriber1_ = self.create_subscription(Image,'image_raw',self.listener_callback,10)
        self.subscriber1_  # prevent unused variable warning
        self.publisher1_ = self.create_publisher(Image, 'hand_controller/image_annotated', 10)
        # Create publisher for velocity command (twist)
        self.publisher2_ = self.create_publisher(Twist, 'hand_controller/cmd_vel', 10)        

        # verbose
        self.declare_parameter("verbose", True)
        self.verbose = self.get_parameter('verbose').value          
        self.get_logger().info('Verbose : "%s"' % self.verbose)

        # use_imshow
        self.declare_parameter("use_imshow", True)
        self.use_imshow = self.get_parameter('use_imshow').value          
        self.get_logger().info('Use imshow : "%s"' % self.use_imshow)
        
        # twist linear.x control
        self.declare_parameter("x_t",  0.0)
        self.declare_parameter("x_a",  0.0)
        self.declare_parameter("x_b", 10.0)
        self.x_t = self.get_parameter('x_t').value
        self.x_a = self.get_parameter('x_a').value
        self.x_b = self.get_parameter('x_b').value
        self.get_logger().info(f'Twist linear.x controls : t={self.x_t}, a={self.x_a}, b={self.x_b}' )

        # twist angular.z control
        self.declare_parameter("z_t",  0.0)
        self.declare_parameter("z_a",  0.0)
        self.declare_parameter("z_b", 10.0)
        self.z_t = self.get_parameter('z_t').value
        self.z_a = self.get_parameter('z_a').value
        self.z_b = self.get_parameter('z_b').value
        self.get_logger().info(f'Twist angular.z controls : t={self.z_t}, a={self.z_a}, b={self.z_b}' )

        # dials_single
        self.declare_parameter("dials_single", True)
        self.dials_single = self.get_parameter('dials_single').value          
        self.get_logger().info('Dials Single mode : "%s"' % self.dials_single)

        # Repo Path
        self.declare_parameter("repo_path", "/root/hand_controller")
        self.repo_path = self.get_parameter('repo_path').value
        self.get_logger().info('Repo path : "%s"' % self.repo_path)

        # Additional Settings (could eventually be mapped to parameters)
        self.bMirrorImage = True
        self.bNormalizedLandmarks = True
        self.bShowDetection = False
        self.bShowLandmarks = True
        
        # Blaze
        sys.path.append(os.path.join(self.repo_path,"blaze_app_python"))
        sys.path.append(os.path.join(self.repo_path,"blaze_app_python","blaze_common"))

        # FOMO
        self.model_default = os.path.join(self.repo_path,"ei_handsv2.eim")
        self.declare_parameter("model", self.model_default)
        self.model = self.get_parameter("model").value

        self.get_logger().info('Edge Impulse FOMO model : "%s"' % self.model)
        self.runner = ImageImpulseRunner(self.model)

        self.model_info = self.runner.init()
        if self.verbose:
            # displays WAY TOO MUCH verbose ... :( ...
            #model_info = runner.init(debug=True) # to get debug print out

            self.get_logger().info('Loaded runner for "%s"' %  self.model_info['project']['owner'] + ' / ' + self.model_info['project']['name'] )
      
        self.labels = self.model_info['model_parameters']['labels'] 
        if self.verbose:
            self.get_logger().info('Labels = %s' %  self.labels )

        self.model_input_width = self.model_info['model_parameters']['image_input_width']
        self.model_input_height = self.model_info['model_parameters']['image_input_height']
        if self.verbose:
            self.get_logger().info('Model Input Size = %d x %d' %  (self.model_input_width, self.model_input_height ))

        # Visual Control Dials

        # Additional Settings (for text overlay)
        self.scale = 1.0
        self.text_fontType = cv2.FONT_HERSHEY_SIMPLEX
        self.text_fontSize = 0.75*self.scale
        self.text_color    = (255,0,0)
        self.text_lineSize = max( 1, int(2*self.scale) )
        self.text_lineType = cv2.LINE_AA
        self.text_x = int(10*self.scale)
        self.text_y = int(30*self.scale)        

        self.get_logger().info("Initialization Successful")


    def listener_callback(self, msg):
        bridge = CvBridge()
        cv2_image = bridge.imgmsg_to_cv2(msg,desired_encoding = "rgb8")
        
        # Mirror horizontally for selfie-mode
        cv2_image = cv2.flip(cv2_image, 1)

        # Process with blaze models.
        image = cv2_image
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        #            
        # Visual Control Dials (init)
        #

        lh_data, rh_data = None, None        

        #            
        # FOMO pipeline
        #

        from visualization import draw_detections
        #from visualization import draw_roi, draw_landmarks
        #from visualization import HAND_CONNECTIONS
    
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        CAMERA_HEIGHT, CAMERA_WIDTH, _ = image.shape
        image_size = max(CAMERA_WIDTH,CAMERA_HEIGHT)
        cropped_size = self.model_input_width    
        img1,scale1,pad1=resize_pad(image,cropped_size,cropped_size)

        # generate features
        features, cropped = self.runner.get_features_from_image(img1)
        cropped = cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR)

        # detection objects (FOMO)
        res = self.runner.classify(features)

        profile_fomo_qty = len(res)
        if profile_fomo_qty > 0:

            # Process results
            if "bounding_boxes" in res["result"].keys():
                if self.verbose:
                    print('Found %d bounding boxes (%f ms, %f ms)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'], res['timing']['classification']))

                for bb in res["result"]["bounding_boxes"]:
                    if self.verbose:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                
                    if bb['label'] == 'face':
                        continue
                    
                    if bb['label'] == 'open':    
                        cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (0, 255, 0), 2)
                    if bb['label'] == 'closed':    
                        cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (0, 0, 255), 2)

                    # Visual Control Dials (prepare hand data)
                    handedness = "Left"
                    x1 = (((bb['x']) / cropped_size) * image_size) - pad1[1]
                    y1 = (((bb['y']) / cropped_size) * image_size) - pad1[0]
                    z1 = 0.0
                    x2 = (((bb['x'] + bb['width']) / cropped_size) * image_size) - pad1[1]
                    y2 = (((bb['y'] + bb['height']) / cropped_size) * image_size) - pad1[0]
                    z2 = 0.0
                    landmarks = np.asarray([[x1,y1,z1],[x2,y2,z2]])
                    if x1 < CAMERA_WIDTH/2:
                        lh_data = HandData(handedness, landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)
                    else:
                        rh_data = HandData(handedness, landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)

                    if bb['label'] == 'open':
                        annotated_image = cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    if bb['label'] == 'closed':    
                        annotated_image = cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Visual Control Dials (display dials)
        if lh_data:
            cv2.circle(annotated_image, (int(lh_data.center_perc[0]*image_width), int(lh_data.center_perc[1]*image_height)), radius=10, color=tria_pink, thickness=-1)  
        if rh_data:
            cv2.circle(annotated_image, (int(rh_data.center_perc[0]*image_width), int(rh_data.center_perc[1]*image_height)), radius=10, color=tria_pink, thickness=-1)
        delta_xy, delta_z = draw_control_overlay(annotated_image, lh_data, rh_data)
        #self.get_logger().info(f"delta_xy = {delta_xy} | delta_z = {delta_z}")
        
        # Create twist message, and publish
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        if self.dials_single:
            # Take both controls from single (left) dial
            linear_x_control = delta_xy[1]
            angular_z_control = delta_xy[0]
        else:
            # Take controls from both dials
            linear_x_control = delta_z[1]
            angular_z_control = delta_xy[0]

        # linear.x control
        if linear_x_control > self.x_t:
            # Advance
            msg.linear.x = self.x_a + (linear_x_control * self.x_b)
        elif linear_x_control < -self.x_t:
            # Back-Up
            msg.linear.x = -self.x_a + (linear_x_control * self.x_b)

        # angular.z control
        if angular_z_control > self.z_t:
            # Turn Left
            msg.angular.z = self.z_a + (angular_z_control * self.z_b)
        elif angular_z_control < -self.z_t:
            # Turn Right
            msg.angular.z = -self.z_a + (angular_z_control * self.z_b)
      
        self.publisher2_.publish(msg)
        if self.verbose:
            self.get_logger().info(f"Published twist msg with linear.x={msg.linear.x:6.3f} angular.z={msg.angular.z:6.3f}")

        if self.use_imshow == True:
            # DISPLAY
            cv2_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('hand_controller_ei1dials_twist_node',cv2_bgr_image)
            cv2.waitKey(1)                    
        
        # CONVERT BACK TO ROS & PUBLISH
        image_ros = bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")        
        self.publisher1_.publish(image_ros)


def main(args=None):
    rclpy.init(args=args)

    hand_controller_ei1dials_twist_node = HandControllerEi1DialsTwistNode()

    rclpy.spin(hand_controller_ei1dials_twist_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hand_controller_ei1dials_twist_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
