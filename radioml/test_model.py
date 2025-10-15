import numpy as np
import onnxruntime as ort
import torch

# Pfade
model_path = "/home/hanna/git/train-radioml/outputs/radioml/model_dynamic_batchsize.onnx"
data_path = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

# Laden des Modells
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Laden der Daten
data = np.load(data_path)
X = data["X"]  # Shape: (samples, 1024, 2)
Y = data["Y"]  # Shape: (samples,)

# Anzahl Samples zum Inferenz-Test
num_samples = 10000

# Zufällige Auswahl von Indizes
np.random.seed(42)
indices = np.random.choice(len(X), size=num_samples, replace=False)
correct = 0
total = 0
for idx in indices:
    x_sample = X[idx]            # (1024, 2)
    y_label = Y[idx]             # int Label

    # Eingabe für das Modell: (1, 1, 1024, 2)
    x_input = np.expand_dims(x_sample, axis=0)  # (1, 1024, 2)
    # x_input = np.expand_dims(x_input, axis=1)   # (1, 1, 1024, 2)

    # ONNX Inferenz (numpy float32 Input)
    pred_onnx = session.run([output_name], {input_name: x_input.astype(np.float32)})[0]  # (1, 24)

    # Prediction Label (argmax)
    pred_label = np.argmax(pred_onnx, axis=1)[0]

    # print(f"Prediction: [{pred_label}]  Ground Truth: [{y_label}]")
    if pred_label == y_label:
        correct = correct + 1
    total = total + 1
print("Accuracy: ", correct / total)



    # 43% accuracy bei model_dynamic_batchsize.onnx, genauso bei model.onnx
    # 57% bei pt (mit eval)

    # wodurch kommt der Unterschied?
