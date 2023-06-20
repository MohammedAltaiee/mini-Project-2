
# first neural network with keras tutorial
# from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
# X = dataset[:,0:8]
# y = dataset[:,8]
# import numpy as np

input = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1]
])

output = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

input_test = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1]
])

output_test = np.array([1, 1, 0, 0, 1, 1, 1, 1])

# Defining the keras model
model = Sequential()
model.add(Dense(12, input_shape=(5,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(input, output, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(input, output)
print('Accuracy: %.2f' % (accuracy*100))


# print(output_predict)
# accuracy = (output_predict == output_test).mean()


output_predict = model.predict(input_test)
rounded_predictions = (output_predict > 0.5).astype(int)
accuracy = np.mean(rounded_predictions == output_test)
print(rounded_predictions)
print('Accuracy: %.2f' % (accuracy*100))