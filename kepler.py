import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense

#importing the dataset
dataset = pd.read_csv('cumulative.csv')

#x is a matrix of the columns to be used in the csv, y is the target label
x = dataset.iloc[:, 4:39].values
y = dataset.iloc[:, 3].values
print(y)
x = pd.DataFrame(x)
x = x.fillna(x.median())


#converts to float
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
#splits the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#initializing Neural Network
classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu",input_dim=35, units=30, kernel_initializer="uniform"))

#adding second hidden layer
classifier.add(Dense(activation="relu",units=30, kernel_initializer="uniform"))

#adding third hidden layer
classifier.add(Dense(activation="relu",units=25, kernel_initializer="uniform"))

#adding fourth hidden layer
classifier.add(Dense(activation="relu",units=20, kernel_initializer="uniform"))

#adding fifth hidden layer
classifier.add(Dense(activation="relu",units=20, kernel_initializer="uniform"))

#output layer
classifier.add(Dense(activation="sigmoid",units=1, kernel_initializer="uniform"))

#compiling neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the model
classifier.fit(x_train, y_train, epochs=200, batch_size=10)

#predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred >= 0.5)

#creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)