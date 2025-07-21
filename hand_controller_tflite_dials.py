'''
Copyright 2025 Tria Technologies Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
#
# Hand Controller with Visual Control Dials
#
# Based on:
#   https://github.com/ljkeller/visual_control/blob/master/proto.py
#
# References:
#   https://www.github.com/AlbertaBeef/blaze_app_python
#   https://www.github.com/AlbertaBeef/asl_mediapipe_pointnet
#
# Dependencies:
#   TFLite
#      tensorflow
#    or
#      tflite_runtime
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

from dataclasses import dataclass
from ctypes import *
from typing import List
import pathlib
#import threading
import time
import sys
import argparse
import glob
import subprocess
import re
import sys

from datetime import datetime

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('blaze_app_python/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_common/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_tflite/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_pytorch/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_vitisai/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_hailo/'))

from blaze_tflite.blazedetector import BlazeDetector as BlazeDetector_tflite
from blaze_tflite.blazelandmark import BlazeLandmark as BlazeLandmark_tflite

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

from timeit import default_timer as timer

bMirrorImage = True
print("[INFO] Mirror Image = ",bMirrorImage)

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


# Parameters (tweaked for video)
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--debug'      , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-z', '--profilelog' , default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-f', '--fps'        , default=False, action='store_true', help="Enable FPS display. Default is off")

args = ap.parse_args()  
  
print('Command line options:')
print(' --debug       : ', args.debug)
print(' --profilelog  : ', args.profilelog)
print(' --withoutview : ', args.withoutview)
print(' --fps         : ', args.fps)


print("[INFO] Searching for USB camera ...")
dev_video = get_video_dev_by_name("uvcvideo")
dev_media = get_media_dev_by_name("uvcvideo")
print(dev_video)
print(dev_media)

if dev_video == None:
    input_video = 0
else:
    input_video = dev_video  

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Open video
cap = cv2.VideoCapture(input_video)
frame_width = CAMERA_WIDTH
frame_height = CAMERA_HEIGHT
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("[INFO] input : camera",input_video," (",frame_width,",",frame_height,")")

# Output directory for captured images
output_dir = './captured-images'
if not os.path.exists(output_dir):
    # Create the output directory if it doesn't already exist      
    os.mkdir(output_dir)            

# Profiling output
profile_csv = './hand_controller_tflite_dials_profiling.csv'
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :",profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :",profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,detections,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,dials,total,fps\n")

pipeline = "hand_controller_tflite_dials"
detector_type = "blazepalm"
landmark_type = "blazehandlandmark"

model1 = "blaze_app_python/blaze_tflite/models/palm_detection_lite.tflite"
blaze_detector = BlazeDetector_tflite(detector_type)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(model1)
 
model2 = "blaze_app_python/blaze_tflite/models/hand_landmark_lite.tflite"
blaze_landmark = BlazeLandmark_tflite(landmark_type)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(model2)

# Visual Control Dials
CV_DRAW_COLOR_PRIMARY = (255, 255, 0)

CONTROL_CIRCLE_DEADZONE_R = 50

CONTROL_CIRCLE_XY_CENTER = (int(CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))
CONTROL_CIRCLE_Z_APERATURE_CENTER = (
    int(3*CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))


@dataclass
class HandData:
    handedness: str
    landmarks: list
    center_perc: tuple

    def __init__(self, handedness, landmarks):
        self.handedness = handedness
        #self.landmarks = landmarks
        #x_avg = sum(lm.x for lm in landmarks) / len(landmarks)
        #y_avg = sum(lm.y for lm in landmarks) / len(landmarks)
        #z_avg = sum(lm.z for lm in landmarks) / len(landmarks)
        self.landmarks = landmarks.copy()
        self.landmarks[:,0] = self.landmarks[:,0] / CAMERA_WIDTH
        self.landmarks[:,1] = self.landmarks[:,1] / CAMERA_HEIGHT        
        landmarks_len = landmarks.shape[0]
        x_avg = sum(self.landmarks[:,0]) / landmarks_len
        y_avg = sum(self.landmarks[:,1]) / landmarks_len
        z_avg = sum(self.landmarks[:,2]) / landmarks_len
        
        self.center_perc = (x_avg, y_avg, z_avg)
        #print(f"center_perc: {self.center_perc}")

def lerp(a, b, t):
    return a(b-a)*t


def draw_control_overlay(img, lh_data=None, rh_data=None):
    # Draw control circle for XY control (left hand)
    cv2.circle(img, CONTROL_CIRCLE_XY_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    if lh_data:
        # Normalize and compute actual pixel position of left hand
        xy_ctl_x_pct_normalized = min((lh_data.center_perc[0] - 0.25) * 4, 1.0)
        xy_ctl_y_pct_normalized = min((lh_data.center_perc[1] - 0.5) * 2, 1.0)

        xy_ctl_x = int(xy_ctl_x_pct_normalized *
                       CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[0]
        xy_ctl_y = int(xy_ctl_y_pct_normalized *
                       CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[1]

        hand_xy_point = (xy_ctl_x, xy_ctl_y)
        center_xy_point = CONTROL_CIRCLE_XY_CENTER

        # Draw line from center to hand position
        cv2.line(img, center_xy_point, hand_xy_point,
                 CV_DRAW_COLOR_PRIMARY, 1)

        # Draw hand position dot
        cv2.circle(img, hand_xy_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    # Draw control circle for Z-aperture (right hand)
    cv2.circle(img, CONTROL_CIRCLE_Z_APERATURE_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    if rh_data:
        z_ctl_pct_normalized = min((rh_data.center_perc[1] - 0.50) * 2, 1.0)
        aperature_ctl_x_pct_normalized = min(
            (rh_data.center_perc[0] - 0.75) * 4, 1.0)

        aperature_ctl_x = int(aperature_ctl_x_pct_normalized *
                              CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[0]
        z_ctl_y = int(z_ctl_pct_normalized *
                      CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[1]

        hand_z_point = (aperature_ctl_x, z_ctl_y)
        center_z_point = CONTROL_CIRCLE_Z_APERATURE_CENTER

        # Draw line from center to hand Z-position
        cv2.line(img, center_z_point, hand_z_point,
                 CV_DRAW_COLOR_PRIMARY, 1)

        # Draw hand position dot
        cv2.circle(img, hand_z_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    # Optional: draw vertical center reference line
    cv2.line(img, (int(CAMERA_WIDTH / 2), 0),
             (int(CAMERA_WIDTH / 2), CAMERA_HEIGHT), CV_DRAW_COLOR_PRIMARY, 1)
       
        
print("================================================================")
print("Hand Controller (TFLite) with Dials")
print("================================================================")
print("\tPress ESC to quit ...")
print("----------------------------------------------------------------")
print("\tPress 'p' to pause video ...")
print("\tPress 'c' to continue ...")
print("\tPress 's' to step one frame at a time ...")
print("\tPress 'w' to take a photo ...")
print("----------------------------------------------------------------")
print("\tPress 'd' to toggle detection overlay on/off")
print("\tPress 'l' to toggle landmarks overlay on/off")
print("\tPress 'e' to toggle scores image on/off")
print("\tPress 'f' to toggle FPS display on/off")
print("\tPress 'z' to toggle profiling log on/off")
print("\tPress 'v' to toggle verbose on/off")
print("----------------------------------------------------------------")
print("\tPress 'm' to toggle horizontal mirror (ie. selfie-mode) ...")
print("\tPress 'n' to toggle use of normalized landmarks for pointnet ...")
print("================================================================")

bStep = False
bPause = False
bWrite = False
bShowDetection = False
bShowLandmarks = True
bShowScores = False
bShowFPS = args.fps
bVerbose = args.debug
bViewOutput = not args.withoutview
bProfileLog = args.profilelog

def ignore(x):
    pass

app_main_title = "Hand Controller Demo"
app_ctrl_title = "Hand Controller Demo"
if bViewOutput:
    cv2.namedWindow(app_main_title)

thresh_min_score = blaze_detector.min_score_thresh
thresh_min_score_prev = thresh_min_score
if bViewOutput:
    cv2.createTrackbar('threshMinScore', app_ctrl_title, int(thresh_min_score*100), 100, ignore)

image = []
output = []

frame_count = 0

# init the real-time FPS counter
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)

while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    frame_count = frame_count + 1

    flag, frame = cap.read()
    if not flag:
        print("[ERROR] cap.read() FAILEd !")
        break

    if bMirrorImage == True:
        # Mirror horizontally for selfie-mode
        frame = cv2.flip(frame, 1)        
        
    # Get trackbar values
    if bViewOutput:
        thresh_min_score = cv2.getTrackbarPos('threshMinScore', app_ctrl_title)
        if thresh_min_score < 10:
            thresh_min_score = 10
            cv2.setTrackbarPos('threshMinScore', app_ctrl_title,thresh_min_score)
        thresh_min_score = thresh_min_score*(1/100)
        if thresh_min_score != thresh_min_score_prev:
            blaze_detector.min_score_thresh = thresh_min_score
            thresh_min_score_prev = thresh_min_score            
                
    #image = cv2.resize(frame,(0,0), fx=scale, fy=scale) 
    image = frame
    output = image.copy()

    #            
    # Visual Control Dials (init hand data)
    #

    lh_data, rh_data = None, None
    
    #
    # Profiling
    #

    profile_resize         = 0
    profile_detector_pre   = 0
    profile_detector_model = 0
    profile_detector_post  = 0
    profile_extract_roi    = 0
    profile_landmark_pre   = 0
    profile_landmark_model = 0
    profile_landmark_post  = 0
    profile_annotate       = 0
    profile_dials          = 0
    #
    profile_total          = 0
    profile_fps            = 0

    #            
    # BlazePalm pipeline
    #
    
    start = timer()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img1,scale1,pad1=blaze_detector.resize_pad(image)
    profile_resize = timer()-start

    normalized_detections = blaze_detector.predict_on_image(img1)
    if len(normalized_detections) > 0:
  
        start = timer()          
        detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
        xc,yc,scale,theta = blaze_detector.detection2roi(detections)
        roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
        profile_extract_roi = timer()-start

        results = blaze_landmark.predict(roi_img)
        #flags, normalized_landmarks = results
        flags, normalized_landmarks, handedness_scores = results
                    
        roi_landmarks = normalized_landmarks.copy()
                
        start = timer() 
        landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]

            if bShowLandmarks == True:
                draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)
                   
        if bShowDetection == True:
            draw_roi(output,roi_box)
            draw_detections(output,detections)
        profile_annotate = timer()-start

        #
        # Visual Control Dials (prepare hand data)
        # 

        start = timer()
        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]

            for i in range(len(flags)):
                flag = flags[i]
                if flag < 0.5:
                   continue

                landmark = landmarks[i]
                handedness_score = handedness_scores[i]
                roi_landmark = roi_landmarks[i,:,:]
                        
                if bMirrorImage == True:
                    if handedness_score >= 0.5:
                        handedness = "Left"
                    else:
                        handedness = "Right"
                else:                               
                    if handedness_score < 0.5:
                        handedness = "Left"
                    else:
                        handedness = "Right"

                # Visual Control Dials (prepare hand data)
                if handedness == "Left":
                    lh_data = HandData(handedness, landmark)
                else:
                    rh_data = HandData(handedness, landmark)

    # Visual Control Dials (display dials)
    draw_control_overlay(output, lh_data, rh_data)

    profile_dials = timer()-start

    # display real-time FPS counter (if valid)
    if rt_fps_valid == True and bShowFPS:
        cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
        if not bViewOutput:
            print("[REAL-TIME]",rt_fps_message)

    #
    # Profiling
    #
    profile_detector_pre   = blaze_detector.profile_pre
    profile_detector_model = blaze_detector.profile_model
    profile_detector_post  = blaze_detector.profile_post
    profile_detector = profile_detector_pre + profile_detector_model + profile_detector_post
    if len(normalized_detections) > 0:
        profile_landmark_pre   = blaze_landmark.profile_pre
        profile_landmark_model = blaze_landmark.profile_model
        profile_landmark_post  = blaze_landmark.profile_post
    profile_landmark = profile_landmark_pre + profile_landmark_model + profile_landmark_post
    profile_total = profile_resize + \
                    profile_detector + \
                    profile_extract_roi + \
                    profile_landmark + \
                    profile_annotate + \
                    profile_dials
    profile_fps = 1.0 / profile_total
    if bProfileLog == True:
        # display profiling results to console
        print(f"[PROFILING] hands={len(normalized_detections)}, FPS={profile_fps:.3f}fps, Total={profile_total*1000:.3f}ms, Detection={profile_detector*1000:.3f}ms, Extract={profile_extract_roi*1000:.3f}ms, Landmark={profile_landmark*1000:.3f}ms, Annotate={profile_annotate*1000:.3f}ms, DIALS={profile_dials*1000:.3f}ms")
        # write profiling results to csv file
        timestamp = datetime.now()
        csv_str = \
            str(timestamp)+","+\
            str(user)+","+\
            str(host)+","+\
            pipeline+","+\
            str(len(normalized_detections))+","+\
            str(profile_resize)+","+\
            str(profile_detector_pre)+","+\
            str(profile_detector_model)+","+\
            str(profile_detector_post)+","+\
            str(profile_extract_roi)+","+\
            str(profile_landmark_pre)+","+\
            str(profile_landmark_model)+","+\
            str(profile_landmark_post)+","+\
            str(profile_annotate)+","+\
            str(profile_dials)+","+\
            str(profile_total)+","+\
            str(profile_fps)+"\n"
        f_profile_csv.write(csv_str)

    #
    # Annotated Output
    #
    
    if bViewOutput:
        # show the output image
        cv2.imshow(app_main_title, output)

    #
    # Keyboard Control
    #
         
    if bStep == True:
        key = cv2.waitKey(0)
    elif bPause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(1)

    #print(key)
    
    bWrite = False
    if key == 119: # 'w'
        bWrite = True

    if key == 115: # 's'
        bStep = True    
    
    if key == 112: # 'p'
        bPause = not bPause

    if key == 99: # 'c'
        bStep = False
        bPause = False

    if key == 100: # 'd'
        bShowDetection = not bShowDetection
        print("[INFO] Show Detection = ",bShowDetection)

    if key == 108: # 'l'
        bShowLandmarks = not bShowLandmarks
        print("[INFO] Show Landmarks = ",bShowLandmarks)             
                
    if key == 101: # 'e'
        bShowScores = not bShowScores
        blaze_detector.display_scores(debug=bShowScores)
        if not bShowScores:
           cv2.destroyWindow("Detection Scores (sigmoid)")

    if key == 102: # 'f'
        bShowFPS = not bShowFPS

    if key == 118: # 'v'
        bVerbose = not bVerbose
        blaze_detector.set_debug(debug=bVerbose) 
        blaze_landmark.set_debug(debug=bVerbose)
        print("[INFO] Verbose = ",bVerbose)

    if key == 122: # 'z'
        bProfileLog = not bProfileLog
        print("[INFO] Profiling = ",bProfileLog)
        
    if key == 109: # 'm'
        bMirrorImage = not bMirrorImage
        print("[INFO] Mirror Image = ",bMirrorImage)
    
    if key == 27 or key == 113: # ESC or 'q':
        break

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = 1
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #print("[INFO] ",rt_fps_message)
        rt_fps_count = 0

# Cleanup
f_profile_csv.close()
cv2.destroyAllWindows()
