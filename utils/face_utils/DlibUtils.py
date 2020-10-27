import numpy as np
import cv2
import dlib

from utils.face_utils.umeyama import umeyama

mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


def get_face_detector():
    return dlib.get_frontal_face_detector()


def get_face_predictor():
    # model_path = '/root/chenpeng/crnn/utils/face_utils/shape_predictor_68_face_landmarks.dat'
    
    model_path = '/root/data/op/utils/face_utils/shape_predictor_68_face_landmarks.dat'
#    model_path = '/home1/chenpeng/crnn-cp/cp/utils/face_utils/shape_predictor_68_face_landmarks.dat'
    return dlib.shape_predictor(model_path)


def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def get_rectangle(points):
    """
    :param points: [N,2]
    :return:
    """
    min_x = np.min(points[:, 0], axis=0)
    min_y = np.min(points[:, 1], axis=0)
    max_x = np.max(points[:, 0], axis=0)
    max_y = np.max(points[:, 1], axis=0)
    return np.array(([min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]))

def get_min_area_rectangle(points):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def generate_mask(img_size, points):
    """
    Get face mask by landmarks
    :param img_size: [W,H]
    :param points: [N,2]
    :return:
    """

    points = cv2.convexHull(points)
    mask = np.zeros(img_size, dtype=float)
    mask = cv2.fillConvexPoly(mask, points, 1)
    return mask


def draw_convex_hull(img, points, color):
    """
    Draw convex hull on original image.
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)
    return img


def align(im, face_detector, lmark_predictor, scale=0):
    im = np.uint8(im)
    faces = face_detector(im, scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for pred in faces:
            points = shape_to_np(lmark_predictor(im, pred))
            trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
            face_list.append([trans_matrix, points])
    return face_list


def get_face_mask(shape, landmarks):
    OVERLAY_POINTS = [
        LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54]
    ]
    im = np.zeros(shape, dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im


def cut_head(img, point, seed=None):
    h, w = img.shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 8
    delta_y = (y2 - y1) / 5
    if seed is not None:
        np.random.seed(seed)
    delta_x = np.random.randint(delta_x)
    delta_y = np.random.randint(delta_y)
    x1_ = np.int(np.maximum(0, x1 - delta_x))
    x2_ = np.int(np.minimum(w - 1, x2 + delta_x))
    y1_ = np.int(np.maximum(0, y1 - delta_y))
    y2_ = np.int(np.minimum(h - 1, y2 + delta_y * 0.5))
    img = img[y1_:y2_, x1_:x2_, :]
    return img
