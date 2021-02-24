import numpy as np
from scipy.io import wavfile
import sounddevice as sd
from scipy.io.wavfile import write
import librosa as lib
import noisereduce as nr
import pickle
from flask import Flask, request, render_template
from scipy.io import wavfile
from pydub import AudioSegment
import os,glob

app = Flask(__name__,template_folder='templates')

model = pickle.load(open('gb_pickle', 'rb'))


def job(file):
    y, rate = lib.load(file)
    mfccs = np.mean(lib.feature.mfcc(y=y, sr=rate, n_mfcc=20).T, axis=0)
    return mfccs

@app.route('/')
def home():
    return render_template('index3.html')


@app.route('/detect',methods=['POST','GET'])
def detect():
    path = r'C:\Users\SUDHEER KONDAPURAM\Desktop\sr'
    
    if request.method=='POST':
        clip,sample_rate=lib.load('noise_free.wav',sr=None)
        if(np.max(clip)<0.03):
            return render_template("index3.html",detect='Result:No music')
        else:
            x=model.predict(job('noise_free.wav').reshape(1,-1))
            if(x[0].any()==0):
                return render_template("index3.html",detect='Result:Music')
            if(x[0].any()==1):
                return render_template("index3.html",detect='Result:Noise')
    return render_template('index3.html')

    
        
@app.route('/record',methods = ['POST', 'GET'])
def record():
    if request.method == 'POST':
        freq = 44100
        duration = 10
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()  # Record audio for the given number of seconds
        # This will convert the NumPy array to an audio
        write("recording1.wav", freq, recording)
        sound, sr = lib.load('recording1.wav')
        dsound, sr = lib.load('dnoise.wav')
        noise = nr.reduce_noise(audio_clip=sound, noise_clip=dsound)
        write('noise_free.wav', sr, noise)
        
    return render_template("index3.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        clip,sample_rate=lib.load('noise_free.wav',sr=None)
        if(np.max(clip)<0.03):
            return render_templatze("index3.html",predict='Result:No music')
        else:
            x=model.predict(job('noise_free.wav').reshape(1,-1))# Convert the NumPy array to audio file
            if(x[0].any()==0):
                return render_template("index3.html",predict='Result:Music')
            if(x[0].any()==1):
                return render_template("index3.html",predict='Result:Noise')
    return render_template('index3.html')

if __name__ == '__main__':
    app.run( )