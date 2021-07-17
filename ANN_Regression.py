import numpy as np
import os
import tensorflow as tf
import keras
from keras import metrics
import pandas as pd
import argparse
from modules.Symmetry_Set_Basis import Symmetry_Set_Basis
from modules.Operator_Basis import Operator_Basis
from datasets.Load_data import Load_data

def Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/kc_house_data.csv")
    parser.add_argument("--model_type", type=str, default="ANN")
    parser.add_argument("--save_dirs", type=str, default="./test1/")
    args = parser.parse_args()
    return args

def main(args):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)

    x_reg, y_reg = Load_data(args.csv_path)
    

    ##================================================= Model Architecture
    inputs = tf.keras.Input(shape=(x_reg.shape[-1]))
    x = inputs
    # x = Symmetry_Set_Basis(num_out=1, rank=tf.rank(x))(x, x, x)
    # x = Operator_Basis(num_out=1,rank=tf.rank(x))(x, x, x)
    # x = Symmetry_Set_Basis(num_out=1, rank=tf.rank(x))(x, x, x)
    # x = Operator_Basis(num_out=1,rank=tf.rank(x))(x, x, x)
    x = Symmetry_Set_Basis(num_out=1, rank=2)(x, x, x)
    x = Operator_Basis(num_out=1, rank=2)(x, x, x)
    x = Symmetry_Set_Basis(num_out=1, rank=2)(x, x, x)
    x = Operator_Basis(num_out=1,rank=2)(x, x, x)
    x = tf.keras.layers.Dense(1)(x)

    # 'ANN'
    modelANN = tf.keras.Model(inputs= inputs, outputs=x, name=args.model_type)
            
    #---------------------------------------------------------- Call Backs
    model_type = args.model_type
    save_dir = args.save_dirs
    model_name = '{}_model.pth'.format(args.model_type) #% model_type 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    import pdb;pdb.set_trace()
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    csv_logger = keras.callbacks.CSVLogger(save_dir+model_type+'.csv')
    earlystop = keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                min_delta=1e-4,
                                patience=3, # 10
                                verbose=1,
                                mode='min', baseline=None, ## 'min' 
                                restore_best_weights=True)
    callbacks = [checkpoint, csv_logger,  earlystop ]

    loss_fn = tf.keras.losses.MeanSquaredError()
    modelANN.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])
    modelANN.summary()


    ##-----------------------------------------Train

    modelANN.fit(np.log(np.abs(x_reg)+1), np.log(np.abs(y_reg)+1), callbacks = callbacks, shuffle=True , epochs=30, batch_size=32, verbose=1) #you need to set validation data


if __name__ == "__main__":
    args = Arguments()
    main(args)

        
        
