#!/bin/env python

import cv2
import os
import numpy as np
import scipy as sp
from PIL import Image
import PIL
import sys

CASCADE_PATH = "haar/haarcascade_frontalface_default.xml"
#FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
FACE_CASCADE = None

def extract_face(img):
    global FACE_CASCADE
    if not FACE_CASCADE:
        FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

    faces = []

    face_coords = FACE_CASCADE.detectMultiScale(img)

    for x, y, w, h in face_coords:
        face = img[y:y+h, x:x+w]
        face_img = Image.fromarray(face, "L")
        new_img = face_img.resize((150, 150), PIL.Image.ANTIALIAS)
        face = np.array(new_img, "uint8")
        faces.append(face)

    return faces


def get_training_data(face_path="yalefaces/"):
    files = os.listdir(face_path)
    faces = []

    for f in files:
        full_path = os.path.join(face_path, f)
        face_class = f.split(".")[0]
        face_id = int(face_class[7:])
        #print full_path, face_id
        img_pil = Image.open(full_path).convert("L")
        img = np.array(img_pil, "uint8")
        found_faces = extract_face(img)
        for face in found_faces:
            faces.append( (face, face_id) )

    return faces


def main(filename, reload=False):

    recognizer = cv2.createEigenFaceRecognizer()
    training_faces = get_training_data()

    face_imgs = [ face for face, _ in training_faces ]
    face_labels = np.array( [ id_ for _, id_ in training_faces ] )

    recognizer.train(face_imgs, face_labels)

    target_img = Image.open(filename).convert("L")
    img = np.array(target_img, "uint8")
    target_faces = extract_face(img)
    for f in target_faces:
        cv2.imshow("Face?", f)
        cv2.waitKey(1000)
        print recognizer.predict(f)


def parse_args(args):
    params = {}
    params['filename'] = args.pop(0)

    while args:
        a = args.pop(0)
        if "a" == "-r":
            params['reload'] = True

    return params

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <filename>"%sys.argv[0]
        sys.exit(-1)

    args = sys.argv[1:]
    params = parse_args(args)

    main(**params)
