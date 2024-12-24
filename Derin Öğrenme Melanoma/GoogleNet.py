import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from keras.src.applications.inception_v3 import InceptionV3
import kagglehub

path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")

print("Path to dataset files:", path)
'''
train_dir='C:\\Users\\Administrator\\Documents\\melanoma_cancer_dataset\\train'
test_dir ='C:\\Users\\Administrator\\Documents\\melanoma_cancer_dataset\\test'
categories = ['benign', 'malignant']
'''

train_dir=os.path.join(path,'melanoma_cancer_dataset','train')
test_dir =os.path.join(path,'melanoma_cancer_dataset','test')
categories = ['benign', 'malignant']

#Veri Arttırma train için
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,          #Normalizasyon
    rotation_range=40,          #Rastgele 40 derece açı ile döndürüyor
    width_shift_range=0.2,      #0.2 genişlikte kaydırıyor
    height_shift_range=0.2,     #0.2 yükseklikte kaydırıyor
    shear_range=0.2,            #kenarlardan kırpma yapar
    zoom_range=0.2,             #0.2 yakınlaştırma
    horizontal_flip=True,       #Görseli Yatayda Döndürme
    fill_mode='nearest',        #Boş alanları en yakın piksel ile dolduruyor
)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,        #Görselin yolu
    target_size=(224,224),      #Görsellerin boyutunu ayarlar
    batch_size=32,              #Her adımda işlenecek adım sayısı
    class_mode='categorical',   #Kategorize modunda sınıflandırır
)

#Veri arttırma test için
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0         #Normalizasyon
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),     #Görsellerin boyutunu ayarlar
    batch_size=32,              #Her adımda işlenecek adım sayısı
    class_mode='categorical',   #Kategorize modunda sınıflandırır
)
GoogleNet = InceptionV3(weights="imagenet",include_top=False, input_shape=(224,224,3))

GoogleNet.trainable= False

#Oluşturulan Model
model = models.Sequential([
    GoogleNet,
    layers.Flatten(),           #Bütün matrisleri düz bir array şekline çevirir.
    layers.Dense(128, activation='relu'),
#Ekstra eklediğim katman
    layers.Dropout(0.2),        #Aşırı öğrenmeyi engellemesi için 0.2 oranında noronları keser
#Ekstra eklediğim katman
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # İki sınıf
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()                 #Modelin özetini yazdırır.

#Modeli eğitmek için
history = model.fit(train_generator, epochs=10, validation_data=test_generator)


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Test için
#image_path = 'C:\\Users\\semii\\OneDrive\\Belgeler\\melanoma_cancer_donusmus\\test\\malignant\\melanoma_10117.jpg'
image_path = os.path.join(test_dir,'Malignant','melanoma_10117.jpg')

img = cv2.imread(image_path)
img = cv2.resize(img,(224,224))
img = img/ 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = np.argmax(prediction)
predicted_label = categories[predicted_class]

if predicted_label == 'benign':
    print("Tahmin Edilen Sınıf: Benign (iyi huylu)")
elif predicted_label == 'malignant':
    print("Tahmin Edilen Sınıf: Malignant (kötü huylu)")
"""# Modeli eğittikten sonra kaydet
model.save('melanoma_classification_model_googlenet.h5')"""
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 inception_v3 (Functional)   (None, 5, 5, 2048)        21802784  
                                                                 
 flatten (Flatten)           (None, 51200)             0         
                                                                 
 dense (Dense)               (None, 128)               6553728   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 28364898 (108.20 MB)
Trainable params: 6562114 (25.03 MB)
Non-trainable params: 21802784 (83.17 MB)
_________________________________________________________________
Epoch 1/10
301/301 [==============================] - 166s 546ms/step - loss: 0.9378 - accuracy: 0.7842 - val_loss: 0.3553 - val_accuracy: 0.8170
Epoch 2/10
301/301 [==============================] - 141s 469ms/step - loss: 0.3614 - accuracy: 0.8344 - val_loss: 0.2788 - val_accuracy: 0.8830
Epoch 3/10
301/301 [==============================] - 139s 463ms/step - loss: 0.3289 - accuracy: 0.8537 - val_loss: 0.2556 - val_accuracy: 0.8850
Epoch 4/10
301/301 [==============================] - 141s 468ms/step - loss: 0.3356 - accuracy: 0.8531 - val_loss: 0.2726 - val_accuracy: 0.8970
Epoch 5/10
301/301 [==============================] - 138s 459ms/step - loss: 0.3201 - accuracy: 0.8590 - val_loss: 0.2644 - val_accuracy: 0.8830
Epoch 6/10
301/301 [==============================] - 141s 467ms/step - loss: 0.3165 - accuracy: 0.8612 - val_loss: 0.2596 - val_accuracy: 0.8920
Epoch 7/10
301/301 [==============================] - 139s 462ms/step - loss: 0.3185 - accuracy: 0.8601 - val_loss: 0.2663 - val_accuracy: 0.8800
Epoch 8/10
301/301 [==============================] - 141s 468ms/step - loss: 0.3079 - accuracy: 0.8670 - val_loss: 0.2589 - val_accuracy: 0.8900
Epoch 9/10
301/301 [==============================] - 139s 462ms/step - loss: 0.3045 - accuracy: 0.8690 - val_loss: 0.3028 - val_accuracy: 0.8540
Epoch 10/10
301/301 [==============================] - 141s 467ms/step - loss: 0.2924 - accuracy: 0.8723 - val_loss: 0.2717 - val_accuracy: 0.8700

Process finished with exit code 0
"""