# Reuters news topics classifier with Keras

import keras
#from keras.datasets import boston_housing
from keras.datasets import reuters

#load the data
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
word_index = reuters.get_word_index()

print ('# of training samples: {}'.format(len(x_train)))
print ('# of test samples: {}'.format(len(x_test)))

num_classes = max(y_train) + 1
print (('# of classes: {}'.format(num_classes)))

index_to_word = {}
for key, value in word_index.items():
	index_to_word[value] =key
print(' '.join(index_to_word[x] for x in x_train[999]))

from keras.preprocessing.text import Tokenizer

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='count')
x_test = tokenizer.sequences_to_matrix(x_test, mode='count')
#note that there are several representation you can choose including 'binary', 'count', 'freq', 'tfidf'
#for tfidf, tokenizer.fit_on_sequences(x_train)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print (x_train.shape), x_train[0]
print (y_train.shape), y_train[0]

#build a model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words, )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print (model.metrics_names)

#train the model
batch_size = 32
epochs = 2
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

#history visualization
import matplotlib.pyplot as plt
#plot traing % validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model train vs validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='center right')
plt.show()

#plot train & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model train vs validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='center right')
plt.show()

#model evaluate using test data
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print ('Test loss: {}'.format(score[0]))
print ('Test accuracy: {}'.format(score[1]))

