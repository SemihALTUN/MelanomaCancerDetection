from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from keras.src.applications.mobilenet import MobileNet

train_dir='C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\train'
test_dir ='C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test'
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
Mobile_Net = MobileNet(weights="imagenet",include_top=False, input_shape=(224,224,3))

Mobile_Net.trainable= False

#Oluşturulan Model
model = models.Sequential([
    Mobile_Net,
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
image_path = 'C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test\\malignant\\melanoma_10117.jpg'

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
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenet_1.00_224 (Functional) │ (None, 7, 7, 1024)     │     3,228,864 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 50176)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │     6,422,656 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 9,659,906 (36.85 MB)
 Trainable params: 6,431,042 (24.53 MB)
 Non-trainable params: 3,228,864 (12.32 MB)
Epoch 1/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 122s 400ms/step - accuracy: 0.8008 - loss: 1.2070 - val_accuracy: 0.8860 - val_loss: 0.2719
Epoch 2/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 131s 437ms/step - accuracy: 0.8624 - loss: 0.3413 - val_accuracy: 0.8950 - val_loss: 0.2574
Epoch 3/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 110s 366ms/step - accuracy: 0.8854 - loss: 0.2983 - val_accuracy: 0.9040 - val_loss: 0.2383
Epoch 4/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 97s 322ms/step - accuracy: 0.8952 - loss: 0.2654 - val_accuracy: 0.9020 - val_loss: 0.2422
Epoch 5/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 83s 276ms/step - accuracy: 0.8904 - loss: 0.2797 - val_accuracy: 0.8950 - val_loss: 0.2596
Epoch 6/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 121s 400ms/step - accuracy: 0.8949 - loss: 0.2574 - val_accuracy: 0.9020 - val_loss: 0.2563
Epoch 7/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 135s 448ms/step - accuracy: 0.8949 - loss: 0.2591 - val_accuracy: 0.9020 - val_loss: 0.2363
Epoch 8/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 133s 442ms/step - accuracy: 0.9020 - loss: 0.2498 - val_accuracy: 0.9050 - val_loss: 0.2646
Epoch 9/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 141s 467ms/step - accuracy: 0.8959 - loss: 0.2663 - val_accuracy: 0.9040 - val_loss: 0.2291
Epoch 10/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 140s 466ms/step - accuracy: 0.9046 - loss: 0.2317 - val_accuracy: 0.9040 - val_loss: 0.2394
'''