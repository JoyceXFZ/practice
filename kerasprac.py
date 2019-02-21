#pip install matplotlib
#pip install keras
import keras 
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Input, Dense, Dropout, Activation 
import matplotlib.pyplot as plt 

model = Sequential()
model.add(Dense(32, activation='relu', input_dim =100))
model.add(Dense(1, activation='sigmoid'))

#summarize model
print(model.summary())

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size = (1000, 1))

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(data, labels, epochs=10, batch_size=32)

plt.imshow(data[0])
plt.show()
#model.save('my_model.h5')
#model = tf.keras.models.load_model('my_model.h5')

