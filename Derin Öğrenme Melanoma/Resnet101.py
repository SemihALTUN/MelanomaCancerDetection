import os.path
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from keras.src.applications.resnet_v2 import ResNet101V2
import kagglehub

# Download latest version
path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")

print("Path to dataset files:", path)

train_dir=os.path.join(path,'melanoma_cancer_dataset','train')
test_dir =os.path.join(path,'melanoma_cancer_dataset','test')
categories = ['benign', 'malignant']

#Veri Arttırma train için
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
)

#Veri arttırma test için
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)
Resnet_101 = ResNet101V2(weights="imagenet",include_top=False, input_shape=(224,224,3))

Resnet_101.trainable= False

#Oluşturulan Model
model = models.Sequential([
    Resnet_101,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
#Ekstra eklediğim katman
    layers.Dropout(0.2),
#Ekstra eklediğim katman
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # İki sınıf
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#Modeli eğiteceğiz
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Test için
image_path = 'C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test\\benign\\melanoma_9607.jpg'
img = cv2.imread(image_path)
img = img / 255.0

img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
predicted_label = categories[predicted_class]

if predicted_label == 'benign':
    print("Tahmin Edilen Sınıf: Benign (iyi huylu)")
elif predicted_label == 'malignant':
    print("Tahmin Edilen Sınıf: Malignant (kötü huylu)")

'''
Epoch 1/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 382s 1s/step - accuracy: 0.7783 - loss: 1.3778 - val_accuracy: 0.8630 - val_loss: 0.4124
Epoch 2/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 299s 993ms/step - accuracy: 0.8507 - loss: 0.4253 - val_accuracy: 0.8800 - val_loss: 0.3080
Epoch 3/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 409s 1s/step - accuracy: 0.8607 - loss: 0.3303 - val_accuracy: 0.8980 - val_loss: 0.2508
Epoch 4/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 420s 1s/step - accuracy: 0.8815 - loss: 0.2943 - val_accuracy: 0.8930 - val_loss: 0.2482
Epoch 5/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 426s 1s/step - accuracy: 0.8869 - loss: 0.2758 - val_accuracy: 0.9020 - val_loss: 0.2489
Epoch 6/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 408s 1s/step - accuracy: 0.8898 - loss: 0.2793 - val_accuracy: 0.8890 - val_loss: 0.2378
Epoch 7/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 414s 1s/step - accuracy: 0.8796 - loss: 0.2843 - val_accuracy: 0.8760 - val_loss: 0.2899
Epoch 8/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 437s 1s/step - accuracy: 0.8898 - loss: 0.2663 - val_accuracy: 0.8960 - val_loss: 0.2374
Epoch 9/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 432s 1s/step - accuracy: 0.8950 - loss: 0.2535 - val_accuracy: 0.9010 - val_loss: 0.2256
Epoch 10/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 422s 1s/step - accuracy: 0.8992 - loss: 0.2422 - val_accuracy: 0.9030 - val_loss: 0.2287
1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step
Tahmin Edilen Sınıf: Benign (iyi huylu)

'''