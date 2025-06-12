import pandas as pd
import numpy as np

def insert_row_into_df(df, row_index, data_dict):
    """
    Inserts values from data_dict into df at the specified row_index.
    - Squeezes numpy arrays and lists
    - Converts 0D arrays to native Python scalars
    - Raises errors on unsupported or non-scalar types
    - Dynamically creates missing columns with appropriate dtypes
    """
    sanitized_dict = {}

    for col, val in data_dict.items():
        # Squeeze if list or array
        if isinstance(val, (np.ndarray, list)):
            val = np.squeeze(val)

        # Convert 0D ndarray to native Python scalar
        if isinstance(val, np.ndarray):
            if val.shape == ():  # scalar-like array
                val = val.item()
            else:
                raise ValueError(
                    f"insert_row_into_df: Value for column '{col}' is still non-scalar after squeeze: {val} (shape: {val.shape})"
                )

        sanitized_dict[col] = val

        # Create column with proper dtype if missing
        if col not in df.columns:
            if isinstance(val, complex):
                df[col] = pd.Series(dtype=np.complex128)
            elif isinstance(val, float):
                df[col] = pd.Series(dtype=float)
            elif isinstance(val, int):
                df[col] = pd.Series(dtype=int)
            elif isinstance(val, str):
                df[col] = pd.Series(dtype=object)
            elif isinstance(val, bool):
                df[col] = pd.Series(dtype=bool)
            elif val is None:
                df[col] = pd.Series(dtype=object)
            else:
                raise TypeError(f"insert_row_into_df: Unsupported value type for column '{col}': {type(val)}")

    # Assign the values to the specified row
    df.loc[row_index, sanitized_dict.keys()] = list(sanitized_dict.values())
    return df
