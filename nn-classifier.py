# ----imports-------
import numpy as np
# from tensorflow import set_random_seed
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
import time
import json

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


def initialize_data():
    image_file_name = r'data/images.npy'
    label_file_name = r'data/labels.npy'
    image_data = np.load(image_file_name)
    label_data = np.load(label_file_name)
    return image_data,label_data


def preprocess_data(image_data, label_data):
    image_data_pre = image_data.reshape(6500,784)  # Each number is represented by 28x28 pixels. 6500 such images
    x_train2, x_test, y_train2, y_test = train_test_split(image_data_pre, label_data, test_size=0.25, stratify=label_data)  # split data train and test
    x_train, x_val, y_train, y_val = train_test_split(x_train2, y_train2, test_size=0.20, stratify=y_train2)
    y_train_cat = keras.utils.to_categorical(y_train)  # Refer: https://keras.io/utils/
    y_val_cat = keras.utils.to_categorical(y_val)
    y_test_cat = keras.utils.to_categorical(y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test, y_train_cat, y_val_cat, y_test_cat


# Refer Keras Documentation for next block: https://keras.io/
def experiments(x, y, x_val, y_val, layers, epochs=10, batch_size=512, optimizer='sgd'):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_cat,
                        validation_data = (x_val, y_val_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    return model,history


def plot(title,xlabel,ylabel,*plottables):
    for plot_ in plottables:
        plt.plot(plot_)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block=False)


image_data, label_data = initialize_data()
x_train, x_val, x_test, y_train, y_val, y_test, y_train_cat, y_val_cat, y_test_cat = preprocess_data(image_data, label_data)
# print(y_train)
# print(y_train_cat)


layers = [
    Dense(512, input_shape=(28*28, ), kernel_initializer='he_normal'),
    Activation('sigmoid'),
    BatchNormalization(),
    Dense(512, kernel_initializer='he_normal'),
    Activation('sigmoid'),
    # Dropout(0.5),
    Dense(10, kernel_initializer='he_normal'),
    Activation('softmax')
]
start = time.time()

model, history = experiments(x_train, y_train_cat, x_val, y_val_cat, layers, 100, 256, keras.optimizers.Adam())

done = time.time()
elapsed = done - start
print(elapsed)
print(history.history['val_acc'][-1])
plot('Model Accuracy', 'accuracy', 'epoch', history.history['acc'], history.history['val_acc'])
# plot('Model Loss','loss','epoch',history.history['loss'],history.history['val_loss'])


# ---------Test and Evaluation Metrics-------
prediction = model.predict(x_test)
y_pred = np.array(prediction).argmax(axis=-1)
print(y_test)
print(y_pred)
cnf_mat = confusion_matrix(y_test, y_pred)
cnf_row, cnf_col = cnf_mat.shape
print(accuracy_score(y_test,y_pred))
df_cm = pd.DataFrame(cnf_mat, range(cnf_row),
                  range(cnf_col))
sn.set(font_scale=2)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
F1_score = 2 * (precision * recall) / (precision + recall)


# ------- Save the model -------------
model.save('my_model.h5')
json_string = model.to_json()

# -------- Save to JSON file ----------
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

# -------- Graphical representation of the model--------
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True)


# image_file_name = r'data/images.npy'
# label_file_name = r'data/labels.npy'
# image_data = np.load(image_file_name)
# label_data = np.load(label_file_name)
#
# image_data_pre = image_data.reshape(6500, 784)
# index = 0
# for image in image_data_pre:
#     index += 1
#     plt.imshow(image.reshape(28, 28), cmap='gray')
#     plt.show(block=False)
#     if index == 1:
#         break
# index = 0
# for label in label_data:
#     index += 1
#     print(label)
#     if index == 1:
#         break
#
#
# x_train2, x_test, y_train2, y_test = train_test_split(image_data_pre, label_data,
#                                                     test_size=0.25,
#                                                     random_state=0,
#                                                     stratify=label_data)
# x_train, x_val, y_train, y_val = train_test_split(x_train2, y_train2,
#                                                     test_size=0.20,
#                                                     random_state=0,
#                                                     stratify=y_train2)
# print(x_train.shape)
# print(x_val.shape)
# print(x_test.shape)
# print(y_train)
# print(keras.utils.to_categorical(y_train))

# y_train_cat = keras.utils.to_categorical(y_train)
# y_val_cat = keras.utils.to_categorical(y_val)
# y_test_cat = keras.utils.to_categorical(y_test)

# model = Sequential()  # declare model
# model.add(Dense(10, input_shape=(28*28, ), kernel_initializer=keras.initializers.he_normal(seed=69)))  # first layer
# model.add(Activation('tanh'))


def recall(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy',recall, precision])

history = model.fit(x_train, y_train_cat,
                    validation_data = (x_val, y_val_cat),
                    epochs=10,
                    batch_size=200,
                    verbose=1)

# import pydot
# import graphviz
# from keras.utils import plot_model
#plot_model(model, to_file='model.png')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)

print(history.history)
prediction = model.predict(x_test)
print(prediction.shape, x_test.shape)
y_pred = np.array(prediction).argmax(axis=-1)
y_pred


print(y_test)
print(y_pred)
cnf_mat = confusion_matrix(y_test, y_pred)
cnf_row, cnf_col = cnf_mat.shape
accuracy_score(y_test, y_pred)
df_cm = pd.DataFrame(cnf_mat, range(cnf_row), range(cnf_col))
sn.set(font_scale=2)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
F1_score = 2 * (precision * recall) / (precision + recall)
print(F1_score)
