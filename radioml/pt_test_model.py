import numpy as np
import torch
from model import Model



params = {
    "model": {
        "num_classes": 24,
        "embedding": {
            "patches": [1, 64],
            "kernel_size": [1, 16],
            "stride": [1, 16],
            "activation": "relu",
            "bits": None
        },
        "positional": "binary",
        "configuration": "original",
        "num_layers": 1,
        "num_heads": 4,
        "emb_dim": 32,
        "expansion_dim": 128,
        "bits": None,
        "activation": "relu",
        "norm": "batch-norm",
        "norm_placement": "post-norm",
        "dropout": 0.0
    }
}


# Pfade
MODEL_PT = "/home/hanna/git/train-radioml/outputs/radioml/model.pt"
DATA_NPZ = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

# Einstellungen
NUM_SAMPLES = 10000
BATCH_SIZE = 256
SEED = 12#42

def main():
    # Daten laden
    data = np.load(DATA_NPZ)
    X = data["X"]  # (N, 1024, 2)
    Y = data["Y"]  # (N,)

    # Zufällige Indizes wählen
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(X), size=min(NUM_SAMPLES, len(X)), replace=False)

    # Modell laden
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(**params["model"])  
    state = torch.load(MODEL_PT, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            x_batch = X[batch_idx]                     # (B, 1024, 2)
            y_batch = Y[batch_idx]                     # (B,)

            # Form auf (B, 1, 1024, 2) bringen wie beim ONNX-Test
            # problem: eine 1 zu viel scheinbar
            x_batch = np.expand_dims(x_batch, axis=1).astype(np.float32)
            x_batch = np.squeeze(x_batch, axis=1) # sonst ist eine dimension zu viel entstanden
            inp = torch.from_numpy(x_batch).to(device)


            out = model(inp)                           # (B, num_classes)
            preds = out.argmax(dim=1).cpu().numpy()

            correct += int((preds == y_batch).sum())
            total += len(y_batch)

    acc = correct / total if total > 0 else 0.0
    print(f"PyTorch Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()




    # Ergebnis: Accuracy mit eigenem Test-File ist auch nur 43%, eventuell sind im eval skript nur wenige daten berücksichtigt?