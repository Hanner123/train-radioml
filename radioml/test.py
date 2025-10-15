import h5py

# Öffne die Datei im Lesemodus
RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"

with h5py.File(RADIOML_PATH, "r") as f:
    # Rekursive Funktion, um die Struktur anzuzeigen
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    f.visititems(print_structure)
