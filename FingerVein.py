#!/usr/bin/env python

import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# =========================
# GLOBALS
# =========================

filename = ""
text = None
canvas = None
app = None

X = Y = None
X_train = X_test = y_train = y_test = None
cnn_model = None

accuracy = []
precision = []
recall = []
fscore = []

labels = ['vein' + str(i) for i in range(1, 31)]
labels = [l.lower() for l in labels]

# =========================
# DATASET UPLOAD (FIXED)
# =========================

def uploadDataset():
    global filename

    filename = filedialog.askdirectory(parent=app)

    text.delete('1.0', END)

    if filename == "":
        text.insert(END, "Dataset not selected\n")
    else:
        text.insert(END, "Dataset Loaded:\n" + filename + "\n")

# =========================
# PREPROCESS (FIXED)
# =========================

def preprocessDataset():

    global X, Y, X_train, X_test, y_train, y_test

    text.delete('1.0', END)

    if filename == "":
        text.insert(END, "Upload dataset first\n")
        return

    X, Y = [], []
    count = 0

    for root_dir, dirs, files in os.walk(filename):

        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):

                path = os.path.join(root_dir, file)

                class_name = os.path.basename(os.path.dirname(path)).lower().strip()

                if class_name not in labels:
                    continue

                label = labels.index(class_name)

                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    continue

                img = cv2.resize(img, (32, 32))

                X.append(img)
                Y.append(label)
                count += 1

    if len(X) == 0:
        text.insert(END, "❌ No images found!\n")
        text.insert(END, "Check dataset structure: dataset/vein1/img.jpg\n")
        return

    X = np.array(X, dtype='float32') / 255.0
    Y = to_categorical(np.array(Y), 30)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    text.insert(END, "✅ Preprocessing Done\n")
    text.insert(END, f"Total Images: {count}\n")
    text.insert(END, f"Train: {len(X_train)} Test: {len(X_test)}\n")

# =========================
# METRICS
# =========================

def calculateMetrics(name, testY, predY):

    global accuracy, precision, recall, fscore

    a = accuracy_score(testY, predY) * 100
    p = precision_score(testY, predY, average='macro') * 100
    r = recall_score(testY, predY, average='macro') * 100
    f = f1_score(testY, predY, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, f"\n{name} Results\n")
    text.insert(END, f"Accuracy: {a:.2f}\n")
    text.insert(END, f"Precision: {p:.2f}\n")
    text.insert(END, f"Recall: {r:.2f}\n")
    text.insert(END, f"F1 Score: {f:.2f}\n")

    cm = confusion_matrix(testY, predY)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name)
    plt.show()

# =========================
# SVM
# =========================

def runSVM():

    if X_train is None:
        text.insert(END, "Preprocess first\n")
        return

    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    yte = np.argmax(y_test, axis=1)

    model = svm.SVC()
    model.fit(Xtr, np.argmax(y_train, axis=1))

    pred = model.predict(Xte)

    calculateMetrics("SVM", yte, pred)

# =========================
# CNN
# =========================

def runCNN():

    global cnn_model

    if X_train is None:
        text.insert(END, "Preprocess first\n")
        return

    cnn_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(30, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    os.makedirs("model", exist_ok=True)
    model_path = "model/cnn_model.h5"

    if not os.path.exists(model_path):
        cnn_model.fit(X_train, y_train,
                      epochs=5,
                      batch_size=16,
                      validation_data=(X_test, y_test))
        cnn_model.save(model_path)
    else:
        cnn_model = load_model(model_path)

    pred = cnn_model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    yte = np.argmax(y_test, axis=1)

    calculateMetrics("CNN", yte, pred)

# =========================
# GRAPH
# =========================

def graph():

    if len(accuracy) < 2:
        text.insert(END, "Run SVM and CNN first\n")
        return

    df = pd.DataFrame([
        ['SVM','Accuracy',accuracy[0]],
        ['CNN','Accuracy',accuracy[1]],
        ['SVM','Precision',precision[0]],
        ['CNN','Precision',precision[1]],
        ['SVM','Recall',recall[0]],
        ['CNN','Recall',recall[1]],
        ['SVM','F1',fscore[0]],
        ['CNN','F1',fscore[1]],
    ], columns=['Model','Metric','Value'])

    df.pivot(index='Metric', columns='Model', values='Value').plot(kind='bar')
    plt.show()

# =========================
# PREDICTION (FULL FIX)
# =========================

def predictDisease(filepath):

    global cnn_model

    if cnn_model is None:
        return "Run CNN first"

    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32,32))
    img = img.astype('float32') / 255.0
    img = img.reshape(1,32,32,3)

    pred = cnn_model.predict(img)

    idx = np.argmax(pred)
    conf = np.max(pred)

    output = cv2.imread(filepath)
    output = cv2.resize(output, (700,300))

    result = f"Identified: {labels[idx]}" if conf > 0.6 else "Not Identified"

    cv2.putText(output, result, (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imwrite("output.png", output)

    return result

# =========================
# PREDICT BUTTON (FIXED NO JUMP ISSUE)
# =========================

def predict():

    filepath = filedialog.askopenfilename(parent=app)

    if filepath == "":
        return

    result = predictDisease(filepath)

    text.insert(END, result + "\n")

    if os.path.exists("output.png"):

        img = Image.open("output.png")
        img = img.resize((700,300))
        img = ImageTk.PhotoImage(img)

        canvas.configure(image=img)
        canvas.image = img

# =========================
# GUI
# =========================

def Main():

    global app, text, canvas

    app = tkinter.Tk()
    app.geometry("1300x800")
    app.title("Finger Vein Recognition System")

    Button(app, text="Upload Dataset", command=uploadDataset).pack()
    Button(app, text="Preprocess", command=preprocessDataset).pack()
    Button(app, text="SVM", command=runSVM).pack()
    Button(app, text="CNN", command=runCNN).pack()
    Button(app, text="Graph", command=graph).pack()
    Button(app, text="Predict", command=predict).pack()

    text = Text(app, height=15, width=120)
    text.pack()

    canvas = Label(app)
    canvas.pack()

    app.mainloop()

if __name__ == "__main__":
    Main()