import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

confidence_threshold = 0.5

# Функция для захвата и сохранения лица
def captureAndSaveFace(frame, face_box, save_path='captured_face.jpg'):
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        face_roi = frame[y1:y2, x1:x2]

        if not face_roi.size == 0:
            face_resized = cv2.resize(face_roi, (1200, 1200))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path, face_gray)
        else:
            print("No face detected.")
    else:
        print("No face detected.")

# Функция для применения размытия заднего фона
def applyBackgroundBlur(frame, face_box, blur_factor=25):
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        height, width, _ = frame.shape

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), blur_factor)
        mask = np.stack([mask] * 3, axis=-1)

        result_img = frame * mask + (1 - mask) * cv2.GaussianBlur(frame, (0, 0), blur_factor)
    else:
        result_img = frame

    return result_img

# Функция для поиска основного лица
def findMainFace(face_boxes):
    if face_boxes:
        main_face = max(face_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return main_face
    else:
        return None

# Функция для подсветки основного лица
def highlightMainFace(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), (104, 117, 123), True, False)
    net.setInput(blob)
    detections = net.forward()

    main_face = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            face_area = (x2 - x1) * (y2 - y1)

            if main_face is None or face_area > main_face[2]:
                main_face = (x1, y1, face_area, x2, y2)

    if main_face is not None:
        x1, y1, _, x2, y2 = main_face
        cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

    return frame_opencv_dnn, [main_face] if main_face is not None else []

# Загрузка модели обнаружения лиц
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
face_net = cv2.dnn.readNet(faceModel, faceProto)

# Загрузка основной модели для распознавания лиц
model = load_model('main_model_for_opencv_ready2.h5')

# Инициализация видеозахвата
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Функция для проверки лица, которое не принадлежит ни к одному классу
def checkUnknownFace(predictions):
    max_confidence = np.max(predictions)
    if max_confidence >= confidence_threshold:
        predicted_class = np.argmax(predictions)
        #print(f"Человек принадлежит к классу: {predicted_class}")
    else:
        print("Это другой человек")

# Функция captureAfterDelay со вставленной проверкой для неизвестного лица
def captureAfterDelay(face_net, model, video):
    capture_start_time = time.time()
    captured_images = []

    while True:
        has_frame, frame = video.read()

        if not has_frame:
            cv2.waitKey()
            break

        result_img, face_boxes = highlightMainFace(face_net, frame)
        main_face = findMainFace(face_boxes)

        if main_face is not None and len(main_face) == 5:
            elapsed_time = time.time() - capture_start_time

            if elapsed_time >= 3 and len(captured_images) < 10:
                x1, y1, _, x2, y2 = main_face
                face_roi = frame[y1:y2, x1:x2]

                if not face_roi.size == 0:
                    face_resized = cv2.resize(face_roi, (224, 224))

                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)

                    face_normalized = face_rgb.astype('float32') / 255.0
                    face_normalized = np.expand_dims(face_normalized, axis=0)

                    predictions = model.predict(face_normalized)
                    captured_images.append(np.argmax(predictions))
                    


            if len(captured_images) == 10:
                most_common_class = max(set(captured_images), key=captured_images.count)
                print(f"Самый частый класс: {most_common_class}")
                captured_images = []
                return True
        cv2.imshow("Face detection", result_img)
        cv2.waitKey(1)

        # Проверка для неизвестного лица
        if main_face is not None and len(main_face) == 5:
            x1, y1, _, x2, y2 = main_face
            face_roi = frame[y1:y2, x1:x2]

            if not face_roi.size == 0:
                face_resized = cv2.resize(face_roi, (224, 224))

                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
                face_normalized = face_rgb.astype('float32') / 255.0
                face_normalized = np.expand_dims(face_normalized, axis=0)

                predictions = model.predict(face_normalized)
                checkUnknownFace(predictions)

    cv2.destroyAllWindows()
    
# Основной код
def main():
    while True:
        has_frame, frame = video.read()

        if not has_frame:
            print("Не удалось получить кадр. Программа завершается.")
            break

        result_img, face_boxes = highlightMainFace(face_net, frame)
        main_face = findMainFace(face_boxes)

        if main_face is not None and len(main_face) == 5:
            x1, y1, _, x2, y2 = main_face
            face_roi = frame[y1:y2, x1:x2]

            if not face_roi.size == 0:
                face_resized = cv2.resize(face_roi, (224, 224))

                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
                face_normalized = face_rgb.astype('float32') / 255.0
                face_normalized = np.expand_dims(face_normalized, axis=0)

                predictions = model.predict(face_normalized)
                max_confidence = np.max(predictions)
                if max_confidence >= confidence_threshold:
                    predicted_class = np.argmax(predictions)
                    print(f"Человек принадлежит к классу: {predicted_class}")

                else:
                    print("Это другой человек")

            cv2.imshow("Face detection", result_img)
            cv2.waitKey(1)

        a = captureAfterDelay(face_net, model, video)
        if a:
            break
main()