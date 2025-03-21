import pandas as pd
import numpy as np

def insert_row_into_df(df, row_index, data_dict):
    """
    Fügt die Werte aus data_dict in df.loc[row_index] ein.
    Dabei werden Spalten dynamisch mit passendem dtype erstellt:
    - np.complex128 falls Wert komplex ist
    - float/int/str/... falls Wert entsprechend.
    """
    for col, val in data_dict.items():
        # Spalte neu anlegen, falls sie noch nicht existiert
        if col not in df.columns:
            if np.iscomplexobj(val):
                # echte Complex-Spalte
                df[col] = pd.Series(dtype=np.complex128)
            elif isinstance(val, float):
                df[col] = pd.Series(dtype=float)
            elif isinstance(val, int):
                df[col] = pd.Series(dtype=int)
            elif isinstance(val, str):
                # Oder pd.StringDtype() falls gewünscht
                df[col] = pd.Series(dtype=object)
            else:
                # Fallback z.B. für bool, NoneType oder andere Objekte
                df[col] = pd.Series(dtype=object)

    # Jetzt existieren alle nötigen Spalten
    # -> Zuweisen der Werte in einer Zeile
    df.loc[row_index, data_dict.keys()] = list(data_dict.values())
    return df