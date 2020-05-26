import tkinter as tk
from tkinter.filedialog import askopenfilename
import pyaudio
import wave
import threading
import pickle
import librosa
import math
import numpy as np


def get_mfcc(file_path):
    y, sr = librosa.load(file_path)  # read .wav file
    hop_length = math.floor(sr * 0.010)  # 10ms hop
    win_length = math.floor(sr * 0.025)  # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # get power
    power = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))
    # mfcc is 13 x T matrix now
    mfcc = np.concatenate([mfcc, power], axis=0)
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 39 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    # return T x 39 (transpose of X)
    return X.T  # hmmlearn use T x N matrix


def predict_model(filepath):
    mfcc = get_mfcc(filepath)
    mfcc = kmeans.predict(mfcc).reshape(-1,1)

    result = 'Not Found'
    value = -1000.0
    for_log = {}
    for key in models:
        model = models[key]
        score = model.score(mfcc, [len(mfcc)])
        for_log[key] = score
        if score > value:
            value = score
            result = key
    print(for_log)
    return result

class UI:
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    frames = []
    filePath = ""
    indexFileRecord = 0
    isRecording = False

    def startrecording(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_format, channels=self.channels, rate=self.fs,
                                  frames_per_buffer=self.chunk, input=True)
        self.isrecording = True

        print('Recording')
        self.t = threading.Thread(target=self.record)
        self.t.start()

    def stoprecording(self):
        self.isrecording = False
        print('recording complete')
        self.filename = self.nameTextBox.get()
        self.filename = self.filename + ".wav"
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.frames.clear()

    def record(self):
        while self.isrecording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def open_file(self):
        self.filePath = askopenfilename()
        self.filePath1.set(self.filePath)

    def predict(self):
        predict_word = predict_model(self.filePath)
        if predict_word == "phai":
            self.predict_text.set("phải")
        elif predict_word == "cach":
            self.predict_text.set("cách")
        elif predict_word == "nguoi":
            self.predict_text.set("người")
        elif predict_word == "benhnhan":
            self.predict_text.set("bệnh nhân")
        elif predict_word == "vietnam":
            self.predict_text.set("Việt Nam")
        else:
            self.predict_text.set(predict_word)

    def __init__(self, master):
        self.btnRecord_text = tk.StringVar()
        self.predict_text = tk.StringVar()
        self.filePath1 = tk.StringVar()

        self.button1 = tk.Button(window, text='rec', command=self.startrecording)
        self.button2 = tk.Button(window, text='stop', command=self.stoprecording)
        self.selectFileButton = tk.Button(window, text="Select File", width=10, command=self.open_file)
        self.predictButton = tk.Button(window, text="Predict", width=10, command=self.predict)
        self.textBeforeFilePath = tk.Label(window, text='File Path: ')
        self.filePathText = tk.Label(window, textvariable=self.filePath1)
        self.textBeforeResult = tk.Label(window, text="Từ đoán được: ")
        self.result = tk.Label(window, textvariable=self.predict_text, width=20)
        self.textBeforeNameTextBox = tk.Label(window, text='File Name: ')
        self.nameTextBox = tk.Entry(window, width=30)

        self.btnRecord_text.set("Start Record")
        self.predict_text.set("")
        self.button1.grid(row=0, column=0, padx=10, pady=10)
        self.button2.grid(row=0, column=1, padx=10, pady=10)
        self.selectFileButton.grid(row=0, column=2, pady=10)
        self.textBeforeNameTextBox.grid(row=1, column=0, pady=10)
        self.nameTextBox.grid(row=1, column=1, pady=20)
        self.predictButton.grid(row=2, column=1, pady=10)
        self.textBeforeFilePath.grid(row=3, column=0, pady=10)
        self.filePathText.grid(row=3, column=1, pady=20)
        self.textBeforeResult.grid(row=4, column=0, pady=10)
        self.result.grid(row=4, column=1, pady=20)


# Load model
models = {}
class_names = ["benhnhan", "vietnam", "cach", "nguoi", "phai"]
for cname in class_names:
    infile = open(cname + '.pkl', 'rb')
    model = pickle.load(infile)
    models[cname] = model
    infile.close()
infile = open('kmeans.pkl', 'rb')
kmeans = pickle.load(infile)
infile.close()

window = tk.Tk()
window.title("Speech Recognition")
window.geometry("500x300")
ui = UI(window)
window.mainloop()