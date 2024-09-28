import os                                                           # Работа с файловой системой
import cv2                                                          # OpenCV для работы с изображениями
import numpy as np                                                  # Библиотека для работы с массивами
import tensorflow as tf                                             # TensorFlow для нейронных сетей
import tensorflow.keras.preprocessing.image as tf_image             # Для генерации отчета о метриках производительности модели.
from sklearn.metrics import classification_report                   # Импортируем classification_report
from tensorflow.keras.callbacks import EarlyStopping                # Ранняя остановка, чтобы избежать переобучения нейросети
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Аугментация изображений
import matplotlib.pyplot as plt                                     # Визуализация данных
from tensorflow.keras.applications import VGG16                     # Импортируем предобученную модель


# ============== 1. Предобработка изображений для обеих категорий ==============

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = preprocess_image(img_path, target_size)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, np.uint8(img * 255))

input_dir_defects = 'data/train/defects/'
output_dir_defects = 'data/preprocessed/defects/'
preprocess_images(input_dir_defects, output_dir_defects)

input_dir_normal = 'data/train/normal/'
output_dir_normal = 'data/preprocessed/normal/'
preprocess_images(input_dir_normal, output_dir_normal)

# ============== 2. Создание модели нейросети ==============

# Загружаем предобученную модель VGG16 без верхней части (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = tf.keras.Sequential([
base_model,  # Используем предобученные сверточные слои VGG16
    tf.keras.layers.Flatten(),  # Преобразуем карты признаков в одномерный вектор
    tf.keras.layers.Dense(512, activation='relu'),  # Добавляем полносвязный слой
    tf.keras.layers.Dense(1, activation='sigmoid')  # Финальный слой для бинарной классификации
])
# Замораживаем слои VGG16, чтобы они не обучались
base_model.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============== 3. Аугментация и загрузка данных ==============
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/preprocessed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/preprocessed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# === Диагностика содержимого батча ===
for data_batch, labels_batch in train_generator:
    print(f"Shape of data batch: {data_batch.shape}")
    print(f"Shape of labels batch: {labels_batch.shape}")
    break

# Ранняя остановка
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ============== 4. Обучение модели ==============

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Построение графиков
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# ============== 5. Оценка модели ==============
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy:.2f}')

# Предсказания и отчёт о метриках
predictions = model.predict(validation_generator)
predicted_classes = np.where(predictions > 0.35, 1, 0)
print(classification_report(validation_generator.labels, predicted_classes))

# ============== 6. Сохранение и загрузка модели ==============
model.save('construction_defect_model.keras')
new_model = tf.keras.models.load_model('construction_defect_model.keras')
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ============== 7. Применение модели для предсказаний ==============

def predict_images(folder_path, model):
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = tf_image.load_img(img_path, target_size=(224, 224))
            img_array = tf_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            if prediction[0] > 0.35:
                print(f'{img_file}: No defect detected')
            else:
                print(f'{img_file}: Defect detected')
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

# Применение к новым изображениям
predict_images('data/inference/images', new_model)

# ============== 8. Визуализация изображений и предсказаний ==============

def plot_images_and_predictions(folder_path, model):
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = tf_image.load_img(img_path, target_size=(224, 224))
            img_array = tf_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            plt.imshow(img)
            plt.axis('off')
            if prediction[0] > 0.35:
                plt.title(f'{img_file}: No defect detected')
            else:
                plt.title(f'{img_file}: Defect detected')
            plt.show()
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

plot_images_and_predictions('data/inference/images', new_model)