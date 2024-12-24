from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from keras.src.applications.resnet_v2 import ResNet50V2

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
Resnet_50 = ResNet50V2(weights="imagenet",include_top=False, input_shape=(224,224,3))

Resnet_50.trainable= False

#Oluşturulan Model
model = models.Sequential([
    Resnet_50,
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
image_path = 'C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test\\malignant\\melanoma_10125.jpg'

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
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ resnet50v2 (Functional)         │ (None, 7, 7, 2048)     │    23,564,800 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 100352)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │    12,845,184 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 36,418,370 (138.93 MB)
 Trainable params: 12,853,570 (49.03 MB)
 Non-trainable params: 23,564,800 (89.89 MB)
Epoch 1/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 251s 820ms/step - accuracy: 0.7892 - loss: 1.6480 - val_accuracy: 0.8780 - val_loss: 0.2953
Epoch 2/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 276s 916ms/step - accuracy: 0.8513 - loss: 0.3543 - val_accuracy: 0.8880 - val_loss: 0.2461
Epoch 3/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 276s 915ms/step - accuracy: 0.8625 - loss: 0.3144 - val_accuracy: 0.8990 - val_loss: 0.2708
Epoch 4/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 236s 785ms/step - accuracy: 0.8835 - loss: 0.2864 - val_accuracy: 0.8870 - val_loss: 0.2743
Epoch 5/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 238s 792ms/step - accuracy: 0.8739 - loss: 0.3245 - val_accuracy: 0.9000 - val_loss: 0.2353
Epoch 6/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 234s 776ms/step - accuracy: 0.8869 - loss: 0.2768 - val_accuracy: 0.8960 - val_loss: 0.2696
Epoch 7/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 236s 783ms/step - accuracy: 0.8820 - loss: 0.2735 - val_accuracy: 0.8960 - val_loss: 0.2347
Epoch 8/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 236s 782ms/step - accuracy: 0.8827 - loss: 0.2812 - val_accuracy: 0.9000 - val_loss: 0.2214
Epoch 9/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 264s 876ms/step - accuracy: 0.8915 - loss: 0.2679 - val_accuracy: 0.9030 - val_loss: 0.2526
Epoch 10/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 207s 687ms/step - accuracy: 0.8905 - loss: 0.2647 - val_accuracy: 0.9070 - val_loss: 0.2242
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 670ms/step

Tahmin Edilen Sınıf: Malignant (Kötü huylu)
'''