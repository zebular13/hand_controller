# Created by Chat-GPT (https://chat.openai.com/)
# using following prompt: write python code for ROS2 publisher node for usb camera at 640x480 resolution
#
# Added automatic detection of USB camera (ie. uvcvideo)

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import glob
import subprocess

def get_media_dev_by_name(src):
    devices = glob.glob("/dev/media*")
    for dev in sorted(devices):
        proc = subprocess.run(['media-ctl','-d',dev,'-p'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

def get_video_dev_by_name(src):
    devices = glob.glob("/dev/video*")
    for dev in sorted(devices):
        proc = subprocess.run(['v4l2-ctl','-d',dev,'-D'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev
                
                
class UsbCamPublisherNode(Node):
    def __init__(self):
        super().__init__('usbcam_publisher_node')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer_ = self.create_timer(1.0 / 30, self.publish_frame)  # 30 FPS
        
        print("[INFO] Searching for USB camera ...")
        dev_video = get_video_dev_by_name("uvcvideo")
        dev_media = get_media_dev_by_name("uvcvideo")
        print(dev_video)
        print(dev_media)

        if dev_video == None:
            input_video = 0
        else:
            input_video = dev_video  

        self.cv_bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(input_video)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def publish_frame(self):
        # Read a frame from the camera
        ret, frame = self.video_capture.read()

        if ret:
            # Convert the frame to sensor_msgs.Image format
            #image_msg = self.cv_bridge.cv2_to_imgmsg(frame) # encoding == U8C3
            #image_msg = self.cv_bridge.cv2_to_imgmsg(frame,encoding='bgr8')
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame_bgr,encoding='rgb8')

            # Publish the image message
            self.publisher_.publish(image_msg)
        else:
            self.get_logger().warn('Failed to read frame from camera.')

def main(args=None):
    rclpy.init(args=args)
    usbcam_publisher_node = UsbCamPublisherNode()
    rclpy.spin(usbcam_publisher_node)

    usbcam_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
