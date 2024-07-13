import os
from PIL import Image
from mtcnn import MTCNN
import cv2

def findMainFace(face_boxes):
    if face_boxes:
        main_face = max(face_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return main_face
    else:
        return None
def crop_and_save_faces(input_folder, output_folder):
    # Создаем выходную папку, если ее нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Инициализация детектора лиц
    detector = MTCNN()

    # Обход всех файлов во входной папке
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Загрузка изображения
            img = cv2.imread(input_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Обнаружение лиц
            faces = detector.detect_faces(img_rgb)

            # Если лицо не обнаружено, пропустить изображение
            if not faces:
                continue

            # Вычисление координат прямоугольной области вокруг лица
            x, y, w, h = faces[0]['box']

            # Обрезка изображения вокруг лица
            face_image = img[y:y+h, x:x+w]

            # Преобразование в черно-белый цвет
            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Создание объекта Image из массива numpy
            pil_image = Image.fromarray(face_image_gray)

            # Изменение размера изображения до нужного
            pil_image = pil_image.resize((224, 224))

            # Сохранение результата
            pil_image.save(output_path)

# Пример использования
input_folder_path = 'dataset/train/populyznov_evgeni'
output_folder_path = 'dataset/train/populyznov_evgeni'
crop_and_save_faces(input_folder_path, output_folder_path)
