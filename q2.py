import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Given training dataset 
input = np.array([[1, 1, 1, 0, 0],[1, 1, 1, 0, 1],[0, 0, 0, 1, 0],[0, 0, 0, 1, 1],[0, 0, 1, 0, 0],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 0, 1, 1, 1],[0, 1, 0, 0, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 0, 1, 1],[0, 1, 1, 0, 0],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[0, 1, 1, 1, 1],[1, 0, 0, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 0],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 0, 1, 1, 1]])
output = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Given testing dataset 
input_test = np.array([[1, 1, 0, 0, 0],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 0, 1, 1],[1, 1, 1, 1, 0],[1, 1, 1, 1, 1]])
output_test = np.array([1, 1, 0, 0, 1, 1, 1, 1])

# Creating the Keras Sequential model
model = Sequential([
    Dense(12, input_shape=(5,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(7, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# Compiling the created model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model by iterating on the data in batches of 50 smaples -- before it was 10 and accuracy for predicted output was 62.50
model.fit(input, output, epochs=150, batch_size=50)

# Evaluating the model
_, accuracy = model.evaluate(input, output)
print('Accuracy: %.2f' % (accuracy*100))

# Making predictions on new data
# Here we use testing data to predict the output and check how accurate it is comparing to the testing output. 
output_predict = model.predict(input_test)
rounded_predictions = (output_predict > 0.5).astype(int)
accuracy = np.mean(rounded_predictions == output_test)

print(rounded_predictions)
print('Accuracy: %.2f' % (accuracy*100))