#coding=utf-8
"""
filename:       digital_recognition.py
Description:
Author:         Dengjie Xiao
IDE:            PyCharm
Change:         2019/7/10  下午10:30    Dengjie Xiao        Create


"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
batch_size = 5#128
num_classes = 10
epochs = 1#20

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# path='mnist.npz'
# f = np.load(path)
# x_train, y_train = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()

x_train = x_train.reshape(60000, 784).astype('float32')
x_train = x_train[0:1000,:]
x_test = x_test.reshape(10000, 784).astype('float32')
x_test = x_test[0:100,:]
# print(x_test.shape())
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_train = y_train[0:1000,:]
y_test = keras.utils.to_categorical(y_test, num_classes)
y_test = y_test[0:100,:]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Total loss on Test Set:', score[0])
print('Accuracy of Testing Set:', score[1])

result = model.predict_classes(x_test)
correct_indices = np.nonzero(result == y_test)[0]
incorrect_indices = np.nonzero(result != y_test)[0]
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(result[correct], y_test[correct]))
plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(result[incorrect], y_test[incorrect]))
    plt.show()







