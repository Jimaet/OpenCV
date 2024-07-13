
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузка обученной модели
model = load_model('main_model_for_opencv.h5')  # Замените 'multi_class_model.h5' на ваш файл модели

# Загрузка изображения для предсказания
img_path = 'dataset/test/1/15371018340.jpg'  # Замените 'path/to/your/test/image.jpg' на путь к изображению, которое вы хотите предсказать
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Нормализация значений пикселей в диапазоне [0, 1]

# Получение предсказания
predictions = model.predict(img_array)

# Вывод результата
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
if predicted_class == 0:
    print("Борис")
elif predicted_class == 1:
    print("Илон Маск")
elif predicted_class == 2:
    print("Саня")   
elif predicted_class == 3:
    print("Путин")     
print(predictions)