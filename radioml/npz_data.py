import h5py
import numpy as np
import os
import yaml
from dataset import RadioMLDataset, get_datasets


def to_npz():
    # Pfade
    hdf5_path = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024_vorverarbeitet.hdf5"
    npz_path = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

    # Anzahl zufälliger Beispiele
    num_samples = 10000
    np.random.seed(42)  # Optional: Für reproduzierbare Ergebnisse

    # HDF5 öffnen
    with h5py.File(hdf5_path, "r") as f:
        total_samples = f["X"].shape[0]
        print("Gesamtanzahl der Samples:", total_samples)

        # Zufällige Indizes ziehen
        random_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        
        # Indizes sortieren (HDF5 verlangt sortierte Indizes!)
        sorted_indices = np.sort(random_indices)
        
        # Daten mit sortierten Indizes lesen
        X_sorted = f["X"][sorted_indices]
        Y_sorted = f["Y"][sorted_indices]

    # Zurück in die ursprüngliche zufällige Reihenfolge bringen
    # (weil X_sorted sortiert ist, aber wir ursprünglich random_indices wollten)
    # → wir finden die Reihenfolge von random_indices in sorted_indices
    inverse_sort = np.argsort(np.argsort(random_indices))  # cleverer Trick
    X = X_sorted[inverse_sort]
    Y = Y_sorted[inverse_sort]

    # X vorbereiten
    X = X.astype(np.float32)

    # Y ggf. one-hot → label
    if Y.ndim == 2 and Y.shape[1] > 1:
        Y = np.argmax(Y, axis=1)
    Y = Y.astype(np.int64)

    # Speichern
    np.savez(npz_path, X=X, Y=Y)

    print(f"{num_samples} zufällige Beispiele gespeichert in {npz_path}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

def vorverarbeitung(use_split=None):
    """
    Erzeugt data/GOLD_XYZ_OSC.0001_1024_vorverarbeitet.hdf5.
    - use_split: None -> gesamtes Dataset (mit Filter/Reshape aus params)
                 "train" / "valid" / "eval" -> exportiert genau diesen Split
                   (nutzt get_datasets mit splits/seed aus params.yaml)
    """
    hdf5_in = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
    hdf5_out = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024_vorverarbeitet.hdf5"
    params_path = "/home/hanna/git/train-radioml/radioml/params.yaml"

    if not os.path.exists(hdf5_in):
        raise FileNotFoundError(f"Input HDF5 not found: {hdf5_in}")

    # lade dataset-params falls vorhanden
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        ds_cfg = params.get("dataset", {})
    else:
        ds_cfg = {}

    snrs = ds_cfg.get("signal_to_noise_ratios", None)
    classes = ds_cfg.get("classes", None)
    reshape = tuple(ds_cfg["reshape"]) if ds_cfg.get("reshape") is not None else None
    # splits = ds_cfg.get("splits", None)
    # seed = ds_cfg.get("seed", 0)

    # Option A: export eines bestimmten splits (train/valid/eval)
    # if use_split in ("train", "valid", "eval"):
    #     assert splits is not None, "splits in params.yaml required for get_datasets"
    #     train, valid, eval_ds = get_datasets(splits=splits, seed=seed, path=hdf5_in,
    #                                          signal_to_noise_ratios=snrs, classes=classes, reshape=reshape)
    #     mapping = {"train": train, "valid": valid, "eval": eval_ds}
    #     ds = mapping[use_split]
    #     # build arrays
    #     X = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
    #     Y = np.stack([ds[i][1] for i in range(len(ds))]).astype(np.int64)
    #     SNR = np.stack([ds[i][2] for i in range(len(ds))]).astype(np.float32)
    # else:
        # Option B: gesamtes (gefiltertes) Dataset via RadioMLDataset
    ds = RadioMLDataset(path=hdf5_in, classes=classes, signal_to_noise_ratios=snrs, reshape=reshape)
    X = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
    Y = np.stack([ds[i][1] for i in range(len(ds))]).astype(np.int64)
    SNR = np.stack([ds[i][2] for i in range(len(ds))]).astype(np.float32)

    os.makedirs(os.path.dirname(hdf5_out), exist_ok=True)
    with h5py.File(hdf5_out, "w") as fout:
        fout.create_dataset("X", data=X, compression="gzip")
        fout.create_dataset("Y", data=Y, compression="gzip")
        fout.create_dataset("SNR", data=SNR, compression="gzip")

    print(f"Vorverarbeitung fertig. Gespeichert: {hdf5_out}")
    print(f"Final samples: {X.shape[0]}, X.shape={X.shape}, Y.shape={Y.shape}")


def shape_of_npz():
    idx = 0
    p = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"
    data = np.load(p)
    X = data["X"] if "X" in data else None
    Y = data["Y"] if "Y" in data else None
    print("File:", p)
    if X is not None:
        print("X shape (all):", X.shape)
        print("X dtype:", X.dtype)
        print(f"X[{idx}] shape:", X[idx].shape)
        print(f"X[{idx}] ndim:", X[idx].ndim)
    else:
        print("No 'X' array in npz")
    if Y is not None:
        print("Y shape (all):", Y.shape)
        print("Y dtype:", Y.dtype)
        try:
            print(f"Y[{idx}] value:", Y[idx])
        except Exception:
            pass

def main():
    # vorverarbeitung()
    # to_npz()
    shape_of_npz()


if __name__ == "__main__":
    main()