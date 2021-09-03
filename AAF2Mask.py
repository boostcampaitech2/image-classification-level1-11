from multiprocessing import Process, Queue
#https://monkey3199.github.io/develop/python/2018/12/04/python-pararrel.html

import os
import dlib
import cv2
import time
import numpy as np
from tqdm import tqdm


source = r'C:\Dataset\All-Age-Faces Dataset\original_images'
target = r'C:\Dataset\All-Age-Faces Dataset\original_images_transfer5070'

## face detector와 landmark predictor 정의
detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 마스크 면을 그릴 landmarks
cls = {}
cls["mask1"] = [2,3,4,5,6,7,8,9,10,11,12,13,14,30]
cls["mask2"] = [3,4,5,6,7,8,9,10,11,12,13,29]
cls["mask3"] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28]
cls["incorrect1"] = [3,4,5,6,7,8,9,10,11,12,13,33]
cls["incorrect2"] = [4,5,6,7,8,9,10,11,12,57]
cls["incorrect3"] = [17,0,1,2,50,52,14,15,16,26]

for name in tqdm(os.listdir(source)):
    id, age = name[:-4].split('A')
    gender = 'female' if int(id) < 7381 else 'male'

    if not(50 <= int(age) <= 70):
        continue

    imagepath = os.path.join(source, name)

    frame = img = cv2.imread(imagepath)
    rects = detector(frame)
    if len(rects) != 1:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) != 1:
            continue
        x, y, w, h = faces[0]
        rects = [dlib.rectangle(int(x), int(y), int(x + w), int(y + h))]
    rect = rects[0]

    targetpath = os.path.join(target, id + "_" + gender + "_Asian_" + age)
    if not os.path.isdir(targetpath):
        os.mkdir(targetpath)

    landmarks = predictor(frame, rect).parts()
    cv2.imwrite(os.path.join(targetpath, "normal.jpg"), frame)
    '''
    for c, l in cls.items():
        frame_target = frame.copy()
        frame_noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
        frame_mask = np.zeros(frame.shape, dtype=np.uint8)
        landmark = [(landmarks[i].x,landmarks[i].y) for i in l]
        frame_mask = cv2.fillPoly(frame_mask, [np.array(landmark, np.int32)], (255, 255, 255))

        cv2.copyTo(frame_noise, frame_mask, frame_target)
        cv2.imwrite(os.path.join(targetpath, c+".jpg"), frame_target)
    '''