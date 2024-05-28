from io import BytesIO
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from flask import Flask, request
from PIL import Image
import base64

app=Flask(__name__)

path = 'Images'
images = []
personNames = []
myList = glob.glob(path+"\*")


for dirpath, dirnames, filenames in os.walk(path):
        print(f'Found directory: {dirpath}')
        for filename in filenames:
            curImg = cv2.imread(f'{dirpath}/{filename}')
            images.append(curImg)
            personNames.append(dirpath.split('\\')[1])
            print(f'\t{filename}')    



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

@app.route('/face_detect',methods=['POST'])
def compare_faces(img_path):  
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    imgS= Image.open(BytesIO(image_data))
    names=[]  
    # imgS=cv2.imread(img_path)
    # imgS=imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    encodeListKnown = findEncodings(images)
    facesCurFrame = face_recognition.face_locations(imgS)#bounding box return
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)#results of facial features

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        indexed_lst = list(enumerate(faceDis))
        sorted_lst = sorted(indexed_lst, key=lambda x: x[1])
        top3_indices = [index for index, value in sorted_lst[:3]]
        names=[personNames[i] for i in  top3_indices]
        distances=[value for index, value in sorted_lst[:3]]
        return {"Names":names,"FaceDistance":distances}

if __name__=='__main__':
    app.run(debug=True)


# print(compare_faces("IMG-20240413-WA0009.jpg"))