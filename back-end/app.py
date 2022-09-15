# Server /app.py
import os
import canciones
from cancionEntity import CancionEntity
import random
import numpy as np
import librosa
import soundfile as sf
from pymongo import MongoClient
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path="/home/murcia/Documentos/javeriana/tesis/Proyecto/Back-end",
            static_folder="/home/murcia/Documentos/javeriana/tesis/Proyecto/Back-end")  # static_folder="/home/murcia/Documentos/javeriana/tesis/Proyecto/Back-end"
app.config['UPLOAD_FOLDER'] = '/home/murcia/Documentos/javeriana/tesis/Proyecto/Back-end/upload/'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
CORS(app)
client = MongoClient("localhost", 27017)
db = client["Tesis"]
index = 0  # 301


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=["GET", "POST"])
def infoPaginaWeb():
    if request.method == "GET":
        global index
        print(index)
        #	indexCancion = random.randint(0, 19)
        objeto = canciones.EXTENDED[index]  # indexCancion
        duracion = canciones.EXTENDED[index]["duracion"]  # - 40 indexCancion
        # timeInit = random.randint(50, duracion)
        timeInit = canciones.EXTENDED[index]["tiempo"]
        index = index + 1
        return {
            "index": objeto["index"],
            "nombre": objeto["nombre"],
            "duracion": objeto["duracion"],
            "tiempo": timeInit
        }
    elif request.method == "POST":
        indexCancion = request.json["numero"]
        timeInit = request.json["tiempo"]
        y, sr = librosa.load("/run/media/dmurcia/Murcia/entrenamiento/" + canciones.EXTENDED[indexCancion]["nombre"],
                             sr=None, offset=timeInit, duration=10.0)
        sf.write("resources/temporal.wav", y, sr)
        response = send_file("resources/temporal.wav", as_attachment=True)
        return response


@app.route('/obtenerArregloBeats', methods=["POST"])
def getArrayBeats():
    indexCancion = request.json["numero"]
    datos = canciones.EXTENDED[indexCancion]
    tempCancion = CancionEntity(info=datos)
    print(tempCancion.getBPM(), tempCancion.getArregloRespuestaEnSegundos())
    return jsonify(tempCancion.getArregloRespuestaEnSegundos())


@app.route('/guardar', methods=["POST"])
def insertar():
    objeto = request.json
    dato = {}
    dato = {
        "index": objeto["index"],
        "nombre": objeto["nombre"],
        "duracion": objeto["duracion"],
        "tiempo": objeto["tiempo"],
        "marcas": objeto["marcas"]
    }

    tabla = db["tbCancionesMarcas"]
    idCreado = tabla.insert_one(dato).inserted_id

    return {
        "codigo": str(idCreado),
        "mensaje": "Creado con Exito!"
    }


@app.route('/prediccion', methods=["POST"])
def prediccion():
    modelo = []
    tipoModelo = int(request.args.get('modelo'))
    inputModelo = []
    if 'file' not in request.files:
        print('No subio ningun archivo')
        return 'No subio ningun archivo'

    file = request.files['file']
    if file.filename == '':
        print('No tiene ningun dato')
        return 'El archivo no tiene ningun dato'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        y, sr = librosa.load(app.config['UPLOAD_FOLDER'] + filename, sr=11025)
        duracion = int(librosa.get_duration(y, sr))

        if duracion >= 140:
            tiempo = 40
            cantidad = 0
            while cantidad < 6:
                y, sr = librosa.load(app.config['UPLOAD_FOLDER'] + filename, sr=11025, offset=tiempo, duration=11.85)
                stft = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048)), ref=np.max)
                melspectogram = librosa.power_to_db(
                    librosa.feature.melspectrogram(S=np.abs(stft) ** 2, sr=sr, n_mels=40), ref=np.max)

                melspectogram = np.abs(melspectogram)
                melspectogram = (melspectogram - np.min(melspectogram)) / np.ptp(melspectogram)
                melspectogram = np.transpose(melspectogram)
                inputModelo.append(melspectogram)
                cantidad = cantidad + 1
                tiempo += 20


        if tipoModelo == 1:
            print("Se escoje el modelo 1")
            modelo = load_model('modelos/modelo1.1')
        elif tipoModelo == 2:
            print("Se escoje el modelo 2")
            modelo = load_model('modelos/modelo2.2')
        elif tipoModelo == 3:
            print("Se escoje el modelo 3")
            modelo = load_model('modelos/modelo3')

        inputModelo = np.array(inputModelo)
        inputModelo = inputModelo.reshape(inputModelo.shape[0], inputModelo.shape[1], inputModelo.shape[2], 1)
        print(inputModelo.shape)
        prediccionParcial = modelo.predict(inputModelo)
        prediccion = np.array([float(0)]*160)
        for i in range(len(prediccionParcial)):
            prediccion += prediccionParcial[i]

        print(np.argmax(prediccion))
        ans = np.argsort(prediccion)[::-1]
        print(ans)
        return jsonify(ans.tolist())




if __name__ == "__main__":
    app.run(host="localhost", port=5000)
