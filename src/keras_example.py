import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf


# #create numpy_array
# numpy_array = np.array([[1, 2], [3, 4], [5,6]])
#
# # convert it to tensorflow
# tensor1 = tf.convert_to_tensor(numpy_array)
# print(tensor1)


iris = load_iris()
X = iris.data
y = iris.target
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

tensor_X_train = tf.convert_to_tensor(scaled_X_train)
tensor_y_train = tf.convert_to_tensor(y_train)

model = Sequential()
model.add(Input(shape=(tensor_X_train.shape[1], )))
model.add(Dense(units=8, input_dim=4, activation='relu'))  # 8 neurons
model.add(Dense(units=8, input_dim=8, activation='relu'))  # 8 neurons
model.add(Dense(units=3, activation='softmax'))  # [0.1,0.3,0.5]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# model.fit(scaled_X_train, y_train, batch_size=32, epochs=150, verbose=2, workers=4)  # one epoch is running through the entire dataset
model.fit(tensor_X_train, tensor_y_train, batch_size=32, epochs=150, verbose=2, workers=4)  # one epoch is running through the entire dataset
predictions = np.argmax(model.predict(scaled_X_test), axis=1)

print(confusion_matrix(y_test.argmax(axis=1), predictions))
print(classification_report(y_test.argmax(axis=1), predictions))
print(accuracy_score(y_test.argmax(axis=1), predictions))
