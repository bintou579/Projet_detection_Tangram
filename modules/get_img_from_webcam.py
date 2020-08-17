"""
Takes a shot every second and splits it in two. Saves the result in a 
folder called 'frames'.

To quit camera mode, press ESC
"""

import cv2
import os
import time
import shutil
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

#labels = {0: 'bateau', 1: 'bol', 2: 'chat', 3: 'coeur', 4: 'cygne', 5: 'lapin', 6: 'maison', 7: 'marteau', 8: 'montagne', 9: 'pont', 10: 'renard', 11: 'tortue'}

labels = ['bateau', 'bol', 'chat', 'coeur', 'cygne', 'lapin', 'maison', 'marteau', 'montagne', 'pont', 'renard', 'tortue']
# Must import model.h5 as model

model_path = "tangram_jason_mobilenetv2.h5"
model = load_model(model_path)

#Change to 1 to get webcam
cam = cv2.VideoCapture(1)

cv2.namedWindow("Camera Shot")

img_counter = 0

res_A = {label:[] for label in labels}
res_B = {label:[] for label in labels}

if not os.path.exists('frames/'):
    os.makedirs('frames/')
else:
    shutil.rmtree('frames/')
    os.makedirs('frames/')

while True:
    start_time = time.time()
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera Shot", frame)

    height, width, dim = frame.shape
    width_cutoff = width // 2
    s1 = frame[:, :width_cutoff]
    s2 = frame[:, width_cutoff:]

    # Resize image to expected size for the model and expansion of dimension from 3 to 4
    dim = (224, 224)
    # s1_up = tf.image.resize(s1/255, (224,224), preserve_aspect_ratio=False)
    s1_up = cv2.resize(s1/255, dim)
    # s1_final = tf.expand_dims(s1_up, axis=0)
    s1_final = s1_up.reshape(1, s1_up.shape[0], s1_up.shape[1], s1_up.shape[2])
    # s2_up = tf.image.resize(s2, (224,224), preserve_aspect_ratio=False)
    s2_up = cv2.resize(s2/255, dim)
    # s2_final = tf.expand_dims(s2_up, axis=0)
    s2_final = s2_up.reshape(1, s2_up.shape[0], s2_up.shape[1], s2_up.shape[2])
    
    # Prediction and creation of results dictionnaries
    result_1 = model.predict(s1_final)
    result_2 = model.predict(s2_final)

    #best_result_A=labels[np.argmax(result_1[0])]
    #best_result_B=labels[np.argmax(result_2[0])]

    top_5_A = result_1[0].argsort()[::-1][:5]
    top_5_B = result_2[0].argsort()[::-1][:5]

    top_l_A = [labels[p] for p in top_5_A]
    top_l_B = [labels[p] for p in top_5_B]

    #Keep up for the dataframe
    for i, label in enumerate(labels):
        res_A[label].append(result_1[0][i])
        res_B[label].append(result_2[0][i])
    end_time = time.time()
    total_fps = 1/(end_time-start_time)
    print("Total time:",end_time-start_time)
    print("FPS:",total_fps, '\n')

    print("Image A")
    print("Best result: ", top_l_A[0], '\n')
    print("Top 5:")
    for i in range(1,5):
        print(i, ": ", top_l_A[i])
    print('\n')
    print("Image B")
    print("Best result: ", top_l_B[0], '\n')
    print("Top 5:")
    for i in range(1,5):
        print(i, ": ", top_l_B[i])
    print('\n\n')

    #Takes a shot every second
    img_name_A = "frames/frame_{}-A.jpg".format(img_counter)
    img_name_B = "frames/frame_{}-B.jpg".format(img_counter)
    cv2.imwrite(img_name_A, s1)
    cv2.imwrite(img_name_B, s2)
    print("{} written!".format(img_name_A.replace("-A.jpg","")))
    
    img_counter += 1
    time.sleep(1)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        

cam.release()

cv2.destroyAllWindows()

df_A = pd.DataFrame(res_A)
df_B = pd.DataFrame(res_B)

df_A.to_csv('results_A.csv')
df_B.to_csv('results_B.csv')

shutil.make_archive('images', 'zip', 'frames/')