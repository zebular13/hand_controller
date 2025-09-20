# Created by Google (https://google.com/)
# using following prompt: ros2 node how to integrate flask to stream image topic

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from flask import Flask, Response
import cv2
import threading
import time

app = Flask(__name__)
bridge = CvBridge()
latest_frame = None
latest_frame_lock = threading.Lock()

class UsbCamSubscriberFlaskNode(Node):
    def __init__(self):
        super().__init__('usbcam_subscriber_flask_node')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.process_video,
            10)
        self.subscription  # prevent unused variable warning

    def process_video(self, msg):
        global latest_frame
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            ret, jpeg = cv2.imencode('.jpg', cv_image)
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

def generate_frames():
    global latest_frame
    while True:
        with latest_frame_lock:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.03) # Adjust for desired frame rate

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def ros2_thread():
    rclpy.init()
    node = UsbCamSubscriberFlaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
def main():    
    ros_thread = threading.Thread(target=ros2_thread)
    ros_thread.start()
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()

