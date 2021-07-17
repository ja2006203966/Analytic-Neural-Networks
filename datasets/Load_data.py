import pandas as pd
import numpy as np

def Load_data(csv_path):
    ## Import data 
    data = pd.read_csv(csv_path)
    x_rg = data.values
    y_rg = x_rg[:,2] # house price
    x_rg = x_rg[:,3:] # others parameters
    x_rg = np.array(x_rg)
    y_rg = np.array(y_rg)
    x_rg = x_rg.astype(np.float32)
    y_rg = y_rg.astype(np.float32)
    return x_rg, y_rg