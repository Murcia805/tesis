import numpy as np
import librosa


def setValue(arreglo, fila, columna, mayorA):
    if (arreglo[fila][columna] < mayorA):
        arreglo[fila][columna] = 0
    else:
        arreglo[fila][columna] = -80


def setDataRespuesta(bpm, inicioTiempo):
    arregloBPM = []
    diferenciaComun = bpm/60
    # Crea el arreglo con los tiempos
    isCalculating = True
    dv = round(inicioTiempo / (bpm/60))
    tiempoBeat = (dv * bpm/60)
    tiempoBeat = tiempoBeat - inicioTiempo
    if tiempoBeat != 0:
        inicioTiempo = ((dv+1) * bpm/60)
        tiempoBeat = 0
        pass

    arregloBPM.append(0)
    while isCalculating:
        arregloBPM.append(tiempoBeat)
        tiempoBeat = tiempoBeat + diferenciaComun
        if tiempoBeat > 11.85:
            isCalculating = False
    #return inicioTiempo, np.array(arregloBPM)

    return inicioTiempo


""""
def setDataRespuesta(arreglo, inicioTiempo, totalTiempo):
    arregloRespuesta = []
    diferenciaComun = 0
    diferenciaComunTemp = abs(arreglo[0] - arreglo[1])
    posiblesDiferencias = []
    isCalculating = True
    isEncontroCadena = False
    i = 2
    cantidad=2

    # Calculo para el arreglo, la posible diferencia de un error maximo de 0.5
    while isCalculating:
        diferenciaTemp = abs(arreglo[i - 1] - arreglo[i])
        cantidad = cantidad + 1
        if diferenciaComunTemp - 0.2 <= diferenciaTemp <= diferenciaComunTemp + 0.2:
            diferenciaComunTemp = (diferenciaComunTemp + diferenciaTemp) / 2
            i = i + 1
            if i >= len(arreglo):
                posiblesDiferencias.append((diferenciaComunTemp, cantidad))
                isCalculating = False
        else:
            if (i - len(arreglo)) >= cantidad:
                cantidad = 2
                posiblesDiferencias.append((diferenciaComunTemp, cantidad))
                i = i + 1
                diferenciaComunTemp = abs(arreglo[i - 1] - arreglo[i])
                i = i + 1
            else:
                isCalculating = False
                diferenciaComun = diferenciaComunTemp

    cnAnterior = 0
    for df, cn in posiblesDiferencias:
        if cn >= cnAnterior:
            cnAnterior = cn
            diferenciaComun = df

    # Crea el arreglo con los tiempos
    isCalculating = True
    dv = inicioTiempo // diferenciaComun
    tiempoBeat = (dv * diferenciaComun) + diferenciaComun
    tiempoBeat = tiempoBeat - inicioTiempo
    while isCalculating:
        arregloRespuesta.append(tiempoBeat)
        tiempoBeat = tiempoBeat + diferenciaComun
        if tiempoBeat > 10:
            isCalculating = False

    # Calculo el BPM de la cancion
    isCalculating = True
    tiempoBeats = 0
    contadorBeats = 0
    while isCalculating:
        if tiempoBeats <= 60 :
            tiempoBeats = tiempoBeats + diferenciaComun
            contadorBeats = contadorBeats + 1
        else:
            isCalculating = False

    return arregloRespuesta, contadorBeats
"""

class CancionEntity():
    def __init__(self, info, offst, sr=11025, res_type="kaiser_best"):
        self.nombre = info["nombre"]

        offst = setDataRespuesta(info["bpm"], offst)

        self.y, self.sr = librosa.load("/run/media/murcia/Murcia/entrenamiento/" + self.nombre, sr=sr, offset=offst,
                                       res_type=res_type, duration=11.85)
        self.stft = librosa.amplitude_to_db(np.abs(librosa.stft(self.y, n_fft=2048)), ref=np.max)

        print(self.stft.shape)

        self.tiempo = [0] * 160 #self.stft.shape[1] Crea el arreglo tiempo del tamaño

        self.bpm = info["bpm"]

        #self.respuestaFrames = librosa.time_to_frames(self.respuestaTiempos, sr=self.sr)

        #for numberFrame in self.respuestaFrames:
        #    self.tiempo[numberFrame] = 1

        if (info["bpm"] >= 60 and info["bpm"] <= 220):
            self.tiempo[info["bpm"] - 60] = 1

        """
        for row in range(0, len(self.stft)):
            for column in range(0, len(self.stft[row])):
                if (row > 0 and row < 110):  # row < 80 or row > 125
                    pass
                    #setValue(self.stft, row, column, -15)
                #elif (row > 85 and row < 110):
                    #pass
                    #setValue(self.stft, row, column, -20)
                else:
                    self.stft[row][column] = -80
        """

        self.melspectogram = librosa.power_to_db(librosa.feature.melspectrogram(S=np.abs(self.stft)**2, sr=sr, n_mels=40, fmin=400, fmax=2048), ref=np.max)

        self.melspectogram = np.abs(self.melspectogram)
        self.melspectogram = (self.melspectogram - np.min(self.melspectogram))/np.ptp(self.melspectogram)
        self.melspectogram = np.transpose(self.melspectogram)

    def getDataArray(self):
        return self.melspectogram

    def getTiempoArray(self):
        return self.tiempo

    def getArregloRespuestasEnFrames(self):
        return self.respuestaFrames

    def getArregloRespuestaEnSegundos(self):
        return self.respuestaTiempos

    def getShape(self):
        return self.melspectogram.shape

    def getBPM(self):
        return self.bpm - 60
