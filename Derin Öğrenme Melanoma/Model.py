from keras.src.legacy.preprocessing.image import ImageDataGenerator #veri arttırma
import numpy as np
from tensorflow.keras import layers, models #model için gerekli
import matplotlib.pyplot as plt #Grafik
import cv2 # görüntü işleme
import os #data set dizini işlemleri için
import kagglehub #kendi sitesinden

path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
print("DataSet Yolu: ", path)

train_dir=os.path.join(path,'melanoma_cancer_dataset','train')
test_dir =os.path.join(path,'melanoma_cancer_dataset','test')
categories = ['benign', 'malignant']

"""
train_dir='C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\train'
test_dir ='C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_dataset\\test'
categories = ['benign', 'malignant']
"""

#Veri arttırma
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, #piksel değerlerini 0-1 aralığına normalize et
    rotation_range=40, #rastgele döndürme
    width_shift_range=0.2, #yatayda kaydırma
    height_shift_range=0.2, #dikeyde kaydırma
    shear_range=0.2, #kenarlardan kırpma
    zoom_range=0.2, #grüntü yakınlaştırma
    horizontal_flip=True, #görüntüleri yatayda rastgele çevirme işlemi
    fill_mode='nearest', #değişim sonrası boş pikselleri en yakın piksel ile doldurma
)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir, #eğitim veri seti
    target_size=(224,224), #görüntüleri yeniden boyutlandırma
    batch_size=32, #her seferinde işlenecek görüntü sayısı
    class_mode='categorical', #çoklu sınıf
)

#Veri Yükleme
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, #normalizasyon
)

test_generator = test_datagen.flow_from_directory(
    test_dir, #test veri set dizini
    target_size=(224, 224), #yeniden boyutlandır
    batch_size=32, #bir kerede işlenecek görüntü
    class_mode='categorical',#çoklu sınıf
)
#Oluşturulan Model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    #f(x)=max(0,x)
    #32 filtre ile Conv2D katmanı
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    #temel özelliklerde kayıp olmaması için ilk droppout 0.3
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),

    layers.Dense(2, activation='softmax')  # sınıflara ait olasılığı 0-1 arası normalize ederek toplamı 1 yapar
                                                #kesin seçimli işlemlerde kullanımı mantıklı olduğu için
])

#Modeli derleme büyük ve gürültülü veri setlerinde daha iyi performans gösterir
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#zorlu ,büyük ve görüntülü data setleri için

model.summary()

#Modeli eğitme işlemi
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# model.save('melanoma_classification_model_mainmodel.h5')

#Test için

#image_path = 'C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test\\malignant\\melanoma_10117.jpg'
image_path = os.path.join(test_dir,'Malignant','melanoma_10117.jpg')

#opencv ile foto okuma
img = cv2.imread(image_path)

img = img / 255.0
img = cv2.resize(img,(224,224)) #Dataseti internetten çekersek modelin eğitildiği boyutta olmadığı için resize ediyoruz
img = np.expand_dims(img, axis=0)#görüntüyü modelin istediği şekle sokar.

#model kullanarak tahmin
prediction = model.predict(img)

#tahmin edilen sınıf indexi
predicted_class = np.argmax(prediction)

#tahmin edilen kategori
predicted_label = categories[predicted_class]

if predicted_label == 'benign':
    print("Tahmin Edilen Sınıf: Benign (iyi huylu)")
elif predicted_label == 'malignant':
    print("Tahmin Edilen Sınıf: Malignant (kötü huylu)")

'''
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 222, 222, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 111, 111, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 111, 111, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 109, 109, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 54, 54, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 54, 54, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 52, 52, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 26, 26, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 26, 26, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 26, 26, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 24, 24, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (None, 12, 12, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 36864)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     9,437,440 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 2)              │           258 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 9,860,930 (37.62 MB)
 Trainable params: 9,859,970 (37.61 MB)
 Non-trainable params: 960 (3.75 KB)

Epoch 1/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 145s 477ms/step - accuracy: 0.7537 - loss: 2.9092 - val_accuracy: 0.6310 - val_loss: 0.8648
Epoch 2/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 117s 390ms/step - accuracy: 0.8024 - loss: 0.9969 - val_accuracy: 0.8300 - val_loss: 0.4060
Epoch 3/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 118s 392ms/step - accuracy: 0.8299 - loss: 0.5089 - val_accuracy: 0.8710 - val_loss: 0.3116
Epoch 4/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 118s 391ms/step - accuracy: 0.8519 - loss: 0.3885 - val_accuracy: 0.8590 - val_loss: 0.3791
Epoch 5/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 119s 395ms/step - accuracy: 0.8543 - loss: 0.3989 - val_accuracy: 0.8790 - val_loss: 0.2753
Epoch 6/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 123s 407ms/step - accuracy: 0.8586 - loss: 0.3880 - val_accuracy: 0.8810 - val_loss: 0.2837
Epoch 7/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 122s 405ms/step - accuracy: 0.8658 - loss: 0.3249 - val_accuracy: 0.8180 - val_loss: 0.4073
Epoch 8/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 124s 413ms/step - accuracy: 0.8657 - loss: 0.3432 - val_accuracy: 0.8590 - val_loss: 0.3104
Epoch 9/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 125s 415ms/step - accuracy: 0.8711 - loss: 0.3218 - val_accuracy: 0.8760 - val_loss: 0.2932
Epoch 10/10
301/301 ━━━━━━━━━━━━━━━━━━━━ 124s 411ms/step - accuracy: 0.8855 - loss: 0.2818 - val_accuracy: 0.8400 - val_loss: 0.3470
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step
Tahmin Edilen Sınıf: Malignant (kötü huylu)

'''