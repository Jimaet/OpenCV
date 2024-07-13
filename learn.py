from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Подготовка данных для обучения и тестирования модели
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Предварительно обученная модель (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Замораживаем большую часть слоев предварительно обученной модели
for layer in base_model.layers:
    layer.trainable = False

# Добавляем новые полносвязные слои для классификации новых данных
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(38, activation='softmax')(x)  # 4 класса

# Создаем модель
model = Model(inputs=base_model.input, outputs=predictions)

# Компилируем модель с выбранными параметрами оптимизации и функцией потерь
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на обучающем наборе и оценка на тестовом
model.fit(train_generator, epochs=120, steps_per_epoch=len(train_generator), validation_data=test_generator)

# Оценка производительности модели на тестовом наборе данных
accuracy = model.evaluate(test_generator)[1]
print("Accuracy on test set: {:.2f}%".format(accuracy * 100))
# Пример сохранения модели с кастомным именем файла
model.save('main_model_for_opencv_ready2.h5')  # Сохранение модели с именем файла 'my_model.h5'