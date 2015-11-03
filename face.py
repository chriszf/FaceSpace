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

    face_coords = FACE_CASCADE.detectMultiScale(img, minNeighbors=9)

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


def get_recognizer(algorithm = "eigen", file_db = "snsd.eigen.yml", source_dir="yalefaces/", reload_ = False):
    algos = {'eigen': cv2.createEigenFaceRecognizer,
             'fisher': cv2.createFisherFaceRecognizer
            }
    recognizer = algos[algorithm]()
    if not reload_ and os.path.exists(file_db):
        recognizer.load(file_db)
        return recognizer

    print "reloading"

    training_faces = get_training_data(source_dir)

    face_imgs = [ face for face, _ in training_faces ]
    face_labels = np.array( [ id_ for _, id_ in training_faces ] )

    recognizer.train(face_imgs, face_labels)
    recognizer.save(file_db)
    return recognizer


def main(filename, reload_=False):

    recognizer = get_recognizer(reload_=reload_) # Default, full set, eigenfaces
    #recognizer = get_recognizer(algorithm = "fisher", file_db = "snsd.fisher.yml", reload_=reload_) # Default, full set, eigenfaces
    #recognizer = get_recognizer(algorithm = "fisher", file_db = "snsd_only.fisher.yml", source_dir="snsd_training/", reload_=reload_) # snsd set, fisher
    #recognizer = get_recognizer(algorithm = "eigen", file_db = "snsd_only.eigen.yml", source_dir="snsd_training/", reload_=reload_) # snsd set, eigenfaces


    target_img = Image.open(filename).convert("L")
    img = np.array(target_img, "uint8")
    target_faces = extract_face(img)
    for f in target_faces:
        cv2.imshow("Face?", f)
        cv2.waitKey(2000)
        print recognizer.predict(f)


def parse_args(args):
    params = {}
    params['filename'] = args.pop(0)

    #params['reload_'] = True
    while args:
        a = args.pop(0)
        if "a" == "-r":
            params['reload_'] = True

    return params

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <filename>"%sys.argv[0]
        sys.exit(-1)

    args = sys.argv[1:]
    params = parse_args(args)

    main(**params)
