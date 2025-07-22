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
# Hand Controller with ASL
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
#   PyTorch
#      torch
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

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

#from blaze_tflite.blazedetector import BlazeDetector as BlazeDetector_tflite
#from blaze_tflite.blazelandmark import BlazeLandmark as BlazeLandmark_tflite
from blaze_hailo.hailo_inference import HailoInference
hailo_infer = HailoInference()
from blaze_hailo.blazedetector import BlazeDetector as BlazeDetector_hailo
from blaze_hailo.blazelandmark import BlazeLandmark as BlazeLandmark_hailo


from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

from timeit import default_timer as timer

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append('./asl_pointnet')
from point_net import PointNet

model_path = './asl_pointnet'
model_name = 'point_net_1.pth'
model = torch.load(os.path.join(model_path, model_name),weights_only=False,map_location=device)            

char2int = {
            "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10, "M":11,
            "N":12, "O":13, "P":14, "Q":15, "R":16, "S":17, "T":18, "U":19, "V":20, "W":21, "X":22, "Y":23
            }

bMirrorImage = True
bNormalizedLandmarks = True
print("[INFO] Mirror Image = ",bMirrorImage)
print("[INFO] Normalized Landmarks = ",bNormalizedLandmarks)

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
profile_csv = './hand_controller_hailo8_asl_profiling.csv'
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :",profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :",profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,detections,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,asl_pre,asl_model,asl_post,total,fps\n")

pipeline = "hand_controller_hailo8_asl"
detector_type = "blazepalm"
landmark_type = "blazehandlandmark"

model1 = "blaze_app_python/blaze_hailo/models/palm_detection_lite.hef"
blaze_detector = BlazeDetector_hailo(detector_type,hailo_infer)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(model1)
 
model2 = "blaze_app_python/blaze_hailo/models/hand_landmark_lite.hef"
blaze_landmark = BlazeLandmark_hailo(landmark_type,hailo_infer)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(model2)
       
        
print("================================================================")
print("Hand Controller (Hailo) with ASL (PyTorch)")
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
    profile_asl_pre        = 0
    profile_asl_model      = 0
    profile_asl_post       = 0
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
        if bShowDetection == True:
            draw_roi(output,roi_box)
            draw_detections(output,detections)
        profile_annotate = timer()-start

        #
        # ASL
        # 

        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]

            for i in range(len(flags)):
                flag = flags[i]
                if flag < 0.5:
                   continue

                start = timer()
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

                if handedness == "Left":
                    hand_x = 10
                    hand_y = 30
                    #hand_color = (0, 0, 255) # BGR : Red
                    hand_color = (0, 255, 0) # BGR : Green
                    #hand_color = (0, 0, 255) # BGR : Blue
                    hand_msg = 'LEFT='
                else:
                    hand_x = frame_width-128
                    hand_y = 30
                    hand_color = (0, 0, 255) # BGR : Red
                    #hand_color = (0, 255, 0) # BGR : Green
                    #hand_color = (255, 0, 0) # BGR : Blue
                    hand_msg = 'RIGHT='

                #print(f"Hand[{i}] flag={flag} handedness={handedness} ...")
                #print("Hand[",i,"]")
                #print("    handedness = ",handedness)
                #print("    landmark = ",landmark)
                #print("    roi_landmark = ",roi_landmark)
                        
                # Determine point cloud of hand
                points_raw=[]
                if bNormalizedLandmarks == True:
                    for lm in roi_landmark:
                        points_raw.append([lm[0], lm[1], lm[2]])
                else:                                
                    for lm in landmark:
                        points_raw.append([lm[0], lm[1], lm[2]])
                points_raw = np.array(points_raw)
                #print("    points_raw=",points_raw)

                # Normalize point cloud of hand
                points_norm = points_raw.copy()
                min_x = np.min(points_raw[:, 0])
                max_x = np.max(points_raw[:, 0])
                min_y = np.min(points_raw[:, 1])
                max_y = np.max(points_raw[:, 1])
                for i in range(len(points_raw)):
                    points_norm[i][0] = (points_norm[i][0] - min_x) / (max_x - min_x)
                    points_norm[i][1] = (points_norm[i][1] - min_y) / (max_y - min_y)
                    # PointNet model was trained on left hands, so need to mirror right hand landmarks
                    if bMirrorImage == True and handedness == "Right":
                        points_norm[i][0] = 1.0 - points_norm[i][0]
                    if bMirrorImage == False and handedness == "Left": # for non-mirrored image
                        points_norm[i][0] = 1.0 - points_norm[i][0]
                #print("    points_norm=",points_norm)
                profile_asl_pre += timer()-start

                start = timer()
                # Draw hand landmarks of each hand.
                if bShowLandmarks == True:                
                    draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, color=hand_color, size=3)
                profile_annotate += timer()-start
                        
                start = timer()
                pointst = torch.tensor([points_norm]).float().to(device)
                label = model(pointst)
                label = label.detach().cpu().numpy()
                profile_asl_model += timer()-start

                start = timer()
                asl_id = np.argmax(label)
                asl_sign = list(char2int.keys())[list(char2int.values()).index(asl_id)]                
                                    
                #asl_text = '['+str(asl_id)+']='+asl_sign
                asl_text = handedness+"="+asl_sign
                #print(asl_text)
                cv2.putText(output,asl_text,
                    (hand_x,hand_y),
                    text_fontType,text_fontSize,
                    hand_color,text_lineSize,text_lineType)        
                profile_asl_post += timer()-start

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
    profile_asl = profile_asl_pre + profile_asl_model + profile_asl_post
    profile_total = profile_resize + \
                    profile_detector + \
                    profile_extract_roi + \
                    profile_landmark + \
                    profile_annotate + \
                    profile_asl
    profile_fps = 1.0 / profile_total
    if bProfileLog == True:
        # display profiling results to console
        print(f"[PROFILING] hands={len(normalized_detections)}, FPS={profile_fps:.3f}fps, Total={profile_total*1000:.3f}ms, Detection={profile_detector*1000:.3f}ms, Extract={profile_extract_roi*1000:.3f}ms, Landmark={profile_landmark*1000:.3f}ms, Annotate={profile_annotate*1000:.3f}ms, ASL={profile_asl*1000:.3f}ms")
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
            str(profile_asl_pre)+","+\
            str(profile_asl_model)+","+\
            str(profile_asl_post)+","+\
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
    
    if key == 110: # 'n'
        bNormalizedLandmarks = not bNormalizedLandmarks        
        print("[INFO] Normalized Landmarks = ",bNormalizedLandmarks)
    
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
