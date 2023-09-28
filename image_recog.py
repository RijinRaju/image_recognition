#image recognition using tensorflow and keras datasets

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
train_images,test_images = train_images/255.0, test_images/255.0

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# model evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc*100:.2f}%")
