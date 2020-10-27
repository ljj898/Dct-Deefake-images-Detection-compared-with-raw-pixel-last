import os
import cv2
import pickle
from tqdm import tqdm
from utils.face_utils.DlibUtils import get_face_predictor, get_face_detector, align
from utils.face_utils.FaceDetector import get_boundingbox
import numpy as np

class FaceProcessor(object):
    def __init__(self, images_path, cache_path):
        self.detector = get_face_detector()
        self.predictor = get_face_predictor()
        self.images_path = images_path
        self.cache_path = cache_path
        face_caches = self.load_cache()
        if face_caches is None:
            face_caches = {}
            count = 0
            for img_path in tqdm(self.images_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                faces = self.detector(img, 0)
                if len(faces) == 0:
                    faces = [None, None]
                    count += 1
                else:
                    landmarks = np.matrix([[p.x, p.y] for p in self.predictor(img, faces[0]).parts()])
                face_caches[img_path] = landmarks
                
            print('Total images of CelebA:{}, Dont detect faces images:{}'.format(len(self.images_path), count))
            self.save_cache(face_caches)

        self.face_caches = face_caches

    def load_cache(self):
        face_caches = None
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                face_caches = pickle.load(f)
        return face_caches

    def save_cache(self, face_caches):
        # Save face and matrix to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(face_caches, f)
