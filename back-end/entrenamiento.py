import json
from random import sample

import canciones
from cancionEntity import CancionEntity
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Input
from tensorflow.keras import layers, Sequential
from tensorflow.python.keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from tensorflow.python.keras.layers import TimeDistributed, BatchNormalization, AveragePooling2D, Concatenate, AveragePooling1D
from tensorflow.keras.models import Model, load_model
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client["Tesis"]

dataset = []

def prediccion(modelo , x, y):
    if modelo == 1:
        model = load_model('modelos/modelo1.1')
        _, tr = model.evaluate(x, y, verbose=0)
        print('Train: %.3f' % (tr))
    elif modelo == 2:
        model = load_model('modelos/modelo2.2')
        _, tr = model.evaluate(x, y, verbose=0)
        print('Train: %.3f' % (tr))
    elif modelo == 3:
        model = load_model('modelos/modelo3')
        _, tr = model.evaluate(x, y, verbose=0)
        print('Train: %.3f' % (tr))


def entrenamiento1(x_train, y_train, x_validate, y_validate):
    num_mels = 40
    tiempo = 256

    clases = 160

    #y_train = to_categorical(y_train)
    #y_validate = to_categorical(y_validate)

    inp = Input(shape=(tiempo, num_mels, 1))

    mdl = BatchNormalization()(inp)
    mdl = Convolution2D(16, (1, 5), activation="elu")(mdl)

    mdl = BatchNormalization()(mdl)
    mdl = Convolution2D(16, (1, 5), activation="elu")(mdl)

    mdl = BatchNormalization()(mdl)
    mdl = Convolution2D(16, (1, 5), activation="elu")(mdl)

    mdl = AveragePooling2D((5, 1))(mdl)
    mdl = BatchNormalization()(mdl)

    conv1 = Convolution2D(24, (1, 32), activation="elu", padding="same")(mdl)
    conv2 = Convolution2D(24, (1, 64), activation="elu", padding="same")(mdl)
    conv3 = Convolution2D(24, (1, 96), activation="elu", padding="same")(mdl)
    conv4 = Convolution2D(24, (1, 128), activation="elu", padding="same")(mdl)
    conv5 = Convolution2D(24, (1, 192), activation="elu", padding="same")(mdl)
    conv6 = Convolution2D(24, (1, 256), activation="elu", padding="same")(mdl)
    cnctnte = Concatenate(axis=1)([conv1, conv2, conv3, conv4, conv5, conv6])
    mdl = Convolution2D(filters=36, kernel_size=1, strides=1, activation="elu")(cnctnte)


    mdl = AveragePooling2D((2, 1))(mdl)
    mdl = BatchNormalization()(mdl)
    conv1 = Convolution2D(24, (1, 32), activation="elu", padding="same")(mdl)
    conv2 = Convolution2D(24, (1, 64), activation="elu", padding="same")(mdl)
    conv3 = Convolution2D(24, (1, 96), activation="elu", padding="same")(mdl)
    conv4 = Convolution2D(24, (1, 128), activation="elu", padding="same")(mdl)
    conv5 = Convolution2D(24, (1, 192), activation="elu", padding="same")(mdl)
    conv6 = Convolution2D(24, (1, 256), activation="elu", padding="same")(mdl)
    cnctnte = Concatenate(axis=1)([conv1, conv2, conv3, conv4, conv5, conv6])
    mdl = Convolution2D(filters=36, kernel_size=1, strides=1, activation="elu")(cnctnte)

    mdl = Flatten()(mdl)

    mdl = BatchNormalization()(mdl)
    mdl = Dropout(0.5)(mdl)
    mdl = Dense(64, activation="elu")(mdl)
    mdl = BatchNormalization()(mdl)
    mdl = Dense(64, activation="elu")(mdl)
    mdl = BatchNormalization()(mdl)

    mdl = Dense(clases, activation="softmax")(mdl)

    # model = keras.Model(inputs, outputs, name="encoder")
    model = Model(inputs=inp, outputs=mdl)
    # binary_crossentropy - categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    # Train the model for 1 epoch from Numpy data
    batch_size = 100
    epochs = 12

    # Train the model for 1 epoch using a dataset
    # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    # print("Fit on Dataset")
    # history = model.fit(dataset, epochs=1)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_validate, y_validate))

    score = model.evaluate(x_validate, y_validate, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    model.save('modelos/modelo1.1')


def entrenamiento2(x_train, y_train, x_validate, y_validate):
    num_mels = 40
    tiempo = 256

    clases = 160

    inp = Input(shape=(tiempo, num_mels, 1))

    model = BatchNormalization(axis=2)(inp)
    model = Convolution2D(32, (3, num_mels), activation="relu")(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D((1, model.shape[2]))(model)

    model = Convolution2D(32, (3, 1), activation="elu")(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 1))(model)

    model = Flatten()(model)
    model = Dense(128, activation="elu")(model)
    model = Dropout(0.25)(model)
    model = Dense(clases, activation="softmax")(model)

    model = Model(inputs=inp, outputs=model)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    batch_size = 20
    epochs = 30

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_validate, y_validate))

    score = model.evaluate(x_validate, y_validate, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    model.save('modelos/modelo2.1')

def entrenamiento3(x_train, y_train, x_validate, y_validate):
    num_mels = 40
    tiempo = 256

    clases = 160

    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(tiempo, num_mels, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))


    model.add(Dense(clases, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    batch_size = 20
    epochs = 70

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_validate, y_validate))

    score = model.evaluate(x_validate, y_validate, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    model.save('modelos/modelo3.1')

def createData():
    data = {}
    data['dta'] = []
    tiempo = 40
    cont = 0
    for i in range(1118):
        if canciones.CANCIONES[i]["bpm"] == 0:
            continue
        bucle = True

        while bucle:

            if tiempo < (canciones.CANCIONES[i]["duracion"] - 40):
                temp_cancion = CancionEntity(info=canciones.CANCIONES[i], offst=tiempo)

                data['dta'].append({
                    "index": i,
                    "in_data": temp_cancion.getDataArray().tolist(),
                    "out_data": temp_cancion.getTiempoArray(),
                    "bpm": temp_cancion.getBPM()
                })
                tiempo = tiempo + 20
                cont = cont + 1
            else:
                bucle = False
                tiempo = 40

        print("Completado: " + str(i+1))

    with open('dataInfo-completo.txt', 'w') as outfile:
        json.dump(data, outfile)


def correccionDatos():
    datos = open("dataInfo-completo.txt")
    datos = json.load(datos)
    indicesAnterior = -1
    indices = []
    cont = 0
    for i in datos['dta']:
        if i['index'] != indicesAnterior:
            cont += 1
            indices.append(i['index'])
            indicesAnterior = i['index']

    porcentaje = math.ceil(cont * 0.10)
    valoresAleatorios = sorted(sample(range(0, cont-1), porcentaje))

    data_train = {}
    data_train['dta'] = []
    data_validation = {}
    data_validation['dta'] = []
    actual = datos['dta'][0]['index']
    data_test = {}
    data_test['dta'] = []
    print(len(datos['dta']))

    for i in range(len(datos['dta'])):
        tmp = datos['dta'][i]
        if tmp['index'] in valoresAleatorios:
            data_test['dta'].append({
                "index": tmp['index'],
                "in_data": tmp['in_data'],
                "bpm": tmp['bpm']
            })
        else:
            if i+1 <= len(datos['dta'])-1:
                print(i+1)
                if actual != datos['dta'][i+1]['index']:
                    actual = datos['dta'][i+1]['index']
                    data_validation['dta'].append({
                        "index": tmp['index'],
                        "in_data": tmp['in_data'],
                        "bpm": tmp['bpm']
                    })
                else:
                    data_train['dta'].append({
                        "index": tmp['index'],
                        "in_data": tmp['in_data'],
                        "bpm": tmp['bpm']
                    })
            else:
                data_validation['dta'].append({
                    "index": tmp['index'],
                    "in_data": tmp['in_data'],
                    "bpm": tmp['bpm']
                })

    with open('dataInfo-train.txt', 'w') as outfile:
        json.dump(data_train, outfile)

    with open('dataInfo-validation.txt', 'w') as outfile:
        json.dump(data_validation, outfile)

    with open('dataInfo-test.txt', 'w') as outfile:
        json.dump(data_test, outfile)



def createInfoCanciones():
    data = {}
    data['canciones'] = []
    for i in range(1118):
        y, sr = librosa.load("/run/media/murcia/Murcia/entrenamiento/cn (" + str(i + 1) + ").mp3")
        data['canciones'].append({
            "index": i,
            "nombre": "cn (" + str(i + 1) + ").mp3",
            "duracion": librosa.get_duration(y, sr)
        })
        print("Completado: " + str(i + 1) + " de 1126")

    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)


def loadData(ubi):
    datos = open(ubi)
    datos = json.load(datos)
    x_datos = []
    y_datos = []
    cont = 0
    for i in datos['dta']:
        cont = cont + 1
        x_datos.append(i['in_data'])
        #y_datos.append(i['out_data'])
        y_datos.append(i['bpm'])

    print(cont)
    return x_datos, y_datos

def main():
    #createData()
    #correccionDatos()

    x, y = loadData("dataInfo-train.txt")
    x_validation, y_validation = loadData("dataInfo-validation.txt")
    #x_test, y_test = loadData("dataInfo-test.txt")
    x = np.array(x)
    y = np.array(y)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    shuffler = np.random.permutation(len(x))
    x = x[shuffler]
    y = y[shuffler]
    shuffler = np.random.permutation(len(x_validation))
    x_validation = x_validation[shuffler]
    y_validation = y_validation[shuffler]
    #shuffler = np.random.permutation(len(x_test))
    #x_test = x_test[shuffler]
    #y_test = y_test[shuffler]
    #entrenamiento1(x, y, x_validation, y_validation)
    entrenamiento2(x, y, x_validation, y_validation)
    #entrenamiento3(x, y, x_validation, y_validation)


if __name__ == '__main__':
    main()

#dataInfo-completo = 8724 datos