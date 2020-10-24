import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import sequential

data = pd.read_csv('finalwithNull.csv')

x = data.drop('Class', axis = 1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)