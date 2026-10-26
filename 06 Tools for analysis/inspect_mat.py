import numpy as np
import scipy.io as sio

FILE_MAT = r"path/to/CVS/PFF/<site>/<ID>/T1/T1_Day1/data.mat"   # <-- set the recording to inspect

def albero(obj, nome, liv=0):
    ind = "    " * liv
    if hasattr(obj, "_fieldnames"):                      # struct
        print(f"{ind}{nome}  [struct]")
        for f in obj._fieldnames:
            albero(getattr(obj, f), f, liv + 1)
    elif isinstance(obj, str) or np.ndim(obj) == 0:      # stringa / scalare
        print(f"{ind}{nome} = {obj}")
    else:                                                # array
        print(f"{ind}{nome}  {np.shape(obj)} {np.asarray(obj).dtype}")

mat = sio.loadmat(FILE_MAT, struct_as_record=False, squeeze_me=True)
for k, v in mat.items():
    if not k.startswith("__"):
        albero(v, k)