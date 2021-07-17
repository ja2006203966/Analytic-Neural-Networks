import pandas as pd
import numpy as np
import os
import glob
import zipfile

def Load_data(csv_path):
    if not os.path.exists(csv_path):
        data_root = os.path.join(*csv_path.split("/")[:-1])
        zip_path = glob.glob(os.path.join(data_root,'*.zip'))[0]
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        
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