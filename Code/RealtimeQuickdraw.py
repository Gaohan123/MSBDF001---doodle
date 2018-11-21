import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os
import ast
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
start = dt.datetime.now()

DP_DIR = 'shuffle_csv/' # set a folder called 'shuffle_csv'
INPUT_DIR = 'data/' # Just put source data into 'data' folder
BASE_SIZE = 256
NCSVS = 100
NCATS = 20
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

dic = {'airplane': 1, 'ambulance': 2, 'bicycle': 3, 'bus': 4, 'canoe': 5, 'car': 6, 'cruise ship': 7,
           'firetruck': 8, 'helicopter': 9, 'hot air balloon': 10, 'motorbike': 11, 'pickup truck': 12,
           'police car': 13, 'sailboat': 14, 'school bus': 15, 'speedboat': 16, 'submarine': 17, 'train': 18,
            'truck': 19}

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(INPUT_DIR)
    return sorted([f2cat(f) for f in files], key=str.lower)

def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

STEPS = 800
EPOCHS = 16
size = 64
batchsize = 680

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

model.load_weights('model_finalweights.h5')

def main():
    emojis = get_QD_emojis()
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110, 50, 50])
    Upper_green = np.array([130, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = 0

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        pred_class, pred_value = keras_predict(model, digit)
                        print(pred_value)

            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            img = overlay(img, emojis[pred_class], 400, 250, 100, 100)
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred = model.predict(processed, batch_size=1, verbose=1)
    pred_value = []
    for value in preds2catids(pred).values[0]:
        pred_value.append(get_keys(dic, value))
    pred_class = preds2catids(pred).values[0][0]
    # pred_class = list(pred_probab).index(max(pred_probab))
    return pred_class, pred_value


def keras_process_image(img):
    image_x = 64
    image_y = 64
    x = np.zeros((1, size, size, 1))
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    # img = np.reshape(img, (-1, image_x, image_y, 1))
    x[0, :, :, 0] = img
    x = preprocess_input(x).astype(np.float32)
    return x


def get_QD_emojis():
    emojis_folder = 'qd_emo/'
    emojis = []
    for i in range(19):
        emojis.append(0)

    for emoji in os.listdir(emojis_folder):
        if emoji == '.DS_Store':
            continue
        print(emoji)
        value = dic[emoji.split('.')[0]]
        emojis[value-1] = cv2.imread(emojis_folder + emoji)
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()