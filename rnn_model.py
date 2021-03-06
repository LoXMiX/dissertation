import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(keras.layers.SimpleRNN(units=128,input_shape=(None, input_dim)))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_hot_train, epochs=20, batch_size=64,validation_data=(x_valid, y_hot_valid))
score = model.evaluate(x_valid, y_hot_valid, batch_size=64)

predictions = model.predict(x_valid)

import sklearn.metrics as metrics
matrix = metrics.confusion_matrix(y_hot_valid.argmax(axis=1), predictions.argmax(axis=1))

matrix