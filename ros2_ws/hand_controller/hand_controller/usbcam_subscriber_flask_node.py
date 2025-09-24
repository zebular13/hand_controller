# Created by Google (https://google.com/)
# using following prompt: ros2 node how to integrate flask to stream image topic

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from flask import Flask, Response, render_template_string, render_template
import cv2
import threading
import time
import os
import socket

app = Flask(__name__,
    template_folder='/root/hand_controller/templates',
    static_folder='/root/hand_controller/static')
bridge = CvBridge()
latest_frame = None
latest_frame_lock = threading.Lock()

# Enable template auto-reload
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Global variable for Flask to access
FLASK_ARGS = {}

class UsbCamSubscriberFlaskNode(Node):
    def __init__(self):
        super().__init__('usbcam_subscriber_flask_node')

        self.declare_parameter('model', 'default_model.eim')
        self.declare_parameter('verbose', False)
        self.declare_parameter('use_imshow', False)
        
        # Get parameter values
        self.model_path = self.get_parameter('model').value
        self.verbose = self.get_parameter('verbose').value
        self.use_imshow = self.get_parameter('use_imshow').value

        self.get_logger().info(f'Model path: {self.model_path}')

        global FLASK_ARGS
        FLASK_ARGS = {
            'model': self.model_path,
            'verbose': self.verbose,
            'use_imshow': self.use_imshow,
            'hostname': socket.gethostname()
        }

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.process_video,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('USB Camera Subscriber Flask Node started')

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
        time.sleep(0.03)  # ~30 FPS

@app.route("/")
def index():
    model_name = os.path.basename(FLASK_ARGS.get('model', 'unknown_model.eim'))
    verbose = FLASK_ARGS.get('verbose', False)
    use_imshow = FLASK_ARGS.get('use_imshow', False)
    hostname = FLASK_ARGS.get('hostname', False)
    return render_template('index.html',
        model_name=model_name,
        verbose=verbose,
        use_imshow=use_imshow,
        hostname=hostname
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': time.time()}

def ros2_thread():
    rclpy.init()
    node = UsbCamSubscriberFlaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main():    
    ros_thread = threading.Thread(target=ros2_thread)
    ros_thread.daemon = True
    ros_thread.start()
    
    # Run with debug mode for auto-reload (but only in development)
    if os.environ.get('FLASK_DEBUG'):
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True, threaded=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()