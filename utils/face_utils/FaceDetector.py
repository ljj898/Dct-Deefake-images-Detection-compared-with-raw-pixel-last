import dlib
import os
import argparse
import cv2
import glob
from tqdm import tqdm

DATASET_PATHS = {
    'original': 'original_sequences',
    'Deepfakes': 'Deepfakes',
    # 'Face2Face': 'Face2Face',
    # 'FaceSwap': 'FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def crop_images(data_path, dataset, compression):
    detector = dlib.get_frontal_face_detector()
    images_dir = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'images')
    croped_img_dir = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'crop13')
    for dir in tqdm(os.listdir(images_dir)):
        os.makedirs(os.path.join(croped_img_dir, dir), exist_ok=True)
        for img_path in glob.glob(os.path.join(images_dir, dir, '*.png')):
            img_name = img_path.split('/')[-1]
            croped_face_path = os.path.join(croped_img_dir, dir, img_name)
            # croped_face = crop_largest_face(detector, predictor, img_path)
            croped_face = crop_face(img_path, detector)
            if croped_face is not None:
                cv2.imwrite(croped_face_path, croped_face)


def crop_face(img_path, detector):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    faces = detector(img, 0)
    if len(faces) != 0:
        x, y, size = get_boundingbox(faces[0], w, h)
        cropped_face = img[y:y + size, x:x + size]
        return cropped_face
    else:
        return None

def crop_face_(img, detector):
    h, w, _ = img.shape
    faces = detector(img, 0)
    if len(faces) != 0:
        x, y, size = get_boundingbox(faces[0], w, h)
        cropped_face = img[y:y + size, x:x + size]
        return cropped_face
    else:
        return None


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str, default='/home1/chenpeng/faceforensics')
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c23')
    args = p.parse_args()

    print(args)

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            crop_images(**vars(args))
    else:
        crop_images(**vars(args))
