import FaceNet
from test_fddb import detect,Mes_img

import os
import cv2
import argparse
import numpy as np
from PIL import Image

def main(arg):
    detection = FaceNet.Detection()
    Encoder = FaceNet.Encoder()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('failed open camara!!!')
    ret,frame = cap.read()
    
    while ret:
        
        faces = detection.find_faces(frame)

        dets,frame = detect(frame)
        frame =Mes_img(dets,frame)
        cv2.imshow('ccc',frame)

        for i,face in enumerate(faces):
            face.embedding = Encoder.generate_embedding(face)
            face.name = arg.name
            if cv2.waitKey(1) & 0xFF == ord('c'):

                new_image = np.transpose(face.image, (1, 2, 0))[:, :, ::-1]
                out = Image.fromarray(new_image)
                out = out.resize((112, 112))
                out = np.asarray(out)
                
                
                # cv2.imshow('crop',out)
                cv2.imwrite('./features_saveimg/%s.jpg'%(face.name),out)
                np.save('./features/%s.npy'%(face.name),face.embedding)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
    

if __name__ == "__main__":
    parses = argparse.ArgumentParser(description="get the name and embedding of faces")
    parses.add_argument('--name',type = str,default='ly',help='the name of face')
    args = parses.parse_args()
    main(args)

