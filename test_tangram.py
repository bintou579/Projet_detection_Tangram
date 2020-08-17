"""
Takes a shot every second and splits it in two. Saves the result in a 
folder called 'frames'.

To quit camera mode, press ESC
"""

import cv2
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import argparse
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--channel_cam", required=False,
	help="choice of the camera : 0 for webcam, 1 for filming device", choices=[0, 1], type = int, default = 1)
ap.add_argument("-s", "--side", required=False,
	help="side of the camera chosen: left or right", choices=['right', 'left'], type = str, default = "left")
ap.add_argument("-i", "--input", required=False,
	help="path to input video", type = str)
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
ap.add_argument("-fps", "--frames_per_second", required = False,
    help="define the number of frames per second", choices=[1, 2, 3, 4], type = int, default = 1)
ap.add_argument("-m", "--model", required = False,
    help="path to the model")

args = ap.parse_args()

output_path = args.output
# "/prediction/photos/video_test/"

if not os.path.exists(args.output):
    os.makedirs(args.output)
else:
    shutil.rmtree(args.output)
    os.makedirs(args.output)

labels = ['bateau', 'bol', 'chat', 'coeur', 'cygne', 'lapin', 'maison', 'marteau', 'montagne', 'pont', 'renard', 'tortue']

model_path = args.model
model = load_model(model_path)

#Change to 1 to get webcam

if args.input is None:
    vid = cv2.VideoCapture(args.channel_cam)
else : 
    input_path = args.input
    vid = cv2.VideoCapture(input_path)

font = cv2.FONT_HERSHEY_SIMPLEX

side = args.side

if vid.isOpened()==False:
    print("Error opening video file")

img_counter = 0

while vid.isOpened():
    start_time = time.time()
    ret, frame = vid.read()
    if not ret:
        print("failed to grab frame")
        break
    

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, dim = frame.shape
    width_cutoff = width // 2
    if side=="left":
        frame=frame[:, :width_cutoff]
    elif side=="right":
        frame= frame[:, width_cutoff:]

    # Resize image to expected size for the model and expansion of dimension from 3 to 4
    dim = (224, 224)
    # s1_up = tf.image.resize(s1/255, (224,224), preserve_aspect_ratio=False)
    frame_up = cv2.resize(frame/255, dim)
    # s1_final = tf.expand_dims(s1_up, axis=0)
    frame_final = frame_up.reshape(1, frame_up.shape[0], frame_up.shape[1], frame_up.shape[2])
    
    # Prediction and creation of results dictionnaries
    result = model.predict(frame_final)

    top_5 = result[0].argsort()[::-1][:5]

    top_l = [labels[p] for p in top_5]

    end_time = time.time()
    total_fps = 1/(end_time-start_time)

    cv2.putText(frame, f"Total time: {end_time-start_time}", 
                (2, 10), font, 0.5, 
                (89, 22, 76), 2,cv2.LINE_4)

    cv2.putText(frame, f"FPS: {total_fps}", 
                (2, 30), font, 0.5, 
                (89, 22, 76), 2,cv2.LINE_4)

    for i, label in enumerate(top_l):
        cv2.putText(frame, f"N{i+1}: {label}", 
                (2, 50+i*15), font, 0.5, 
                (89, 22, 76), 2,cv2.LINE_4)

    cv2.imshow("Tangram", frame)

    img_name= os.path.join(args.output, "frame_{}.jpg".format(img_counter))
    cv2.imwrite(img_name, frame)

    img_counter += 1
    time.sleep(1/args.frames_per_second)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        

vid.release()

cv2.destroyAllWindows()