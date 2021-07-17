import numpy as np
import os
import tensorflow as tf
import keras
from keras import metrics
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)
        
## Import data
data = pd.read_csv("./kc_house_data.csv")
x_rg = data.values
y_rg = x_rg[:,2] # house price
x_rg = x_rg[:,3:] # others parameters
x_rg = np.array(x_rg)
y_rg = np.array(y_rg)
x_rg = x_rg.astype(np.float32)
y_rg = y_rg.astype(np.float32)

## Customer Class
class Symmetry_Set_Basis(tf.keras.layers.Layer):
    def __init__(self, node=7, num_out=1, rank=2):
        super(Symmetry_Set_Basis, self).__init__()
        self.node = node
        self.wq = tf.keras.layers.Dense(node)
        self.wq2 = tf.keras.layers.Dense(node)
        self.wk = tf.keras.layers.Dense(node)
        self.num_out = num_out
        self.p = [[0,2,1],[0,1,3,2], [0,1,2,4,3], [0,1,2,3,5,4]][rank-2]
        self.rui = tf.random_uniform_initializer(minval=-10, maxval=10)
    
    def Tile_reshape(self, cn):
        a = cn.shape
        b = tf.zeros(tf.rank(cn))+1
        b = tf.cast(b,tf.int32)
        a = tf.concat([b[:-1],b[-1:]*tf.constant(a[-1], tf.int32)], -1)
        return a
    def VP(self, m, cn): # m: order,  cn: input tensor, k: range
        vp = tf.math.pow(cn,m)
        vp = tf.reduce_sum(vp, axis = -1)
        vp = tf.expand_dims(vp, axis = -1)
        vp = tf.tile(vp, self.Tile_reshape(cn))
        return vp
    
    def VC1(self, cn):
        
        vc = tf.reduce_sum(cn, axis = -1)
        vc = tf.expand_dims(vc, axis=-1)
        vc = tf.tile(vc, self.Tile_reshape(cn))
        return vc
    def VC2(self, cn):
        vc = (self.VC1(cn)**2 - self.VP(2, cn))/2
        return vc
    def VC3(self, cn):
        vc1 = self.VC1(cn)
        vp2 = self.VP(2,cn)
        vp3 = self.VP(3,cn)
        vc = (vc1**3-vp3-3*(vp2 * vc1-vp3 ))/6
        return vc
    def VC4(self, cn):
        vc = (self.VC3(cn)*self.VP(1,cn) - self.VC2(cn)*self.VP(2,cn) + self.VC1(cn)*self.VP(3,cn) - self.VP(4,cn) )/4
        return vc
    
    def call(self, q, k, v):
        vc1 = self.VC1(v)
        vc2 = self.VC2(v)
        vc3 = self.VC3(v)
        vc4 = self.VC4(v)
        vp2 = self.VP(2,v)
        vp3 = self.VP(3,v)
        vp4 = self.VP(4,v)
##-------------------------------------------------------- this block is to prevent divergence
        vc2 = tf.math.pow(tf.math.abs(vc2),1/2)*tf.math.sign(vc2)
        vc3 = tf.math.pow(tf.math.abs(vc3),1/3)*tf.math.sign(vc3)
        vc4 = tf.math.pow(tf.math.abs(vc4),1/4)*tf.math.sign(vc4)
        vp2 = tf.math.pow(tf.math.abs(vp2),1/2)*tf.math.sign(vp2)
        vp3 = tf.math.pow(tf.math.abs(vp3),1/3)*tf.math.sign(vp3)
        vp4 = tf.math.pow(tf.math.abs(vp4),1/4)*tf.math.sign(vp4)
##----------------------------------------------------------
        vc1 = tf.expand_dims(vc1, axis=-1)
        vc2 = tf.expand_dims(vc2, axis=-1)
        vc3 = tf.expand_dims(vc3, axis=-1)
        vc4 = tf.expand_dims(vc4, axis=-1)
        vp2 = tf.expand_dims(vp2, axis=-1)
        vp3 = tf.expand_dims(vp3, axis=-1)
        vp4 = tf.expand_dims(vp4, axis=-1)
        
        v = tf.concat([vc1, vc2, vc3, vc4, vp2, vp3, vp4], axis =-1)
        q = tf.expand_dims(q, axis=-1)
        q = self.wq(q)
        q = tf.transpose(q, perm=self.p) 
        k = self.wk(v)
        k = tf.transpose(k, perm=self.p) 
        k = k/tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.math.pow(k,2) ,axis=-1))+1e-10, axis=-1)
        q = q/tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.math.pow(q,2) ,axis=-1))+1e-10, axis=-1)
        n = tf.math.multiply_no_nan(k,q)
        n = tf.transpose(n, perm=self.p)
        v = tf.math.multiply_no_nan(n,v)
        v = tf.reduce_sum(v, axis=-1)
        return v
        
class Operator_Basis(tf.keras.layers.Layer):
    def __init__(self, node=3, num_out=1, rank=2):
        super(Operator_Basis, self).__init__()
        self.node = node
        self.wq = tf.keras.layers.Dense(node)
        self.wq2 = tf.keras.layers.Dense(node)
        self.wk = tf.keras.layers.Dense(node)
        self.alpha = tf.keras.layers.Dense(1)
        self.num_out = num_out
        self.p = [[0,2,1],[0,1,3,2], [0,1,2,4,3], [0,1,2,3,5,4]][rank-2]
    
    def call(self, q, k, v):
        sqrt = tf.math.sqrt(tf.math.abs(v)+1e-10)
        ln = tf.math.log(tf.math.abs(v)+1)
#         exp = tf.math.exp(v)
        rgsn = self.alpha(tf.expand_dims(v, axis=-1))
        sqrt= tf.expand_dims(sqrt, axis=-1)
        ln = tf.expand_dims(ln, axis=-1)
#         exp = tf.expand_dims(exp, axis=-1)
#         v = tf.concat([sqrt, ln, exp, rgsn], axis =-1)
        v = tf.concat([sqrt, ln, rgsn], axis =-1)

        q = tf.expand_dims(q, axis=-1)
        q = self.wq(q)
        q = tf.transpose(q, perm=self.p) 
        k = self.wk(v)
        k = tf.transpose(k, perm=self.p) 
        
        k = k/tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.math.pow(k,2)+1e-10 ,axis=-1)), axis=-1)
        q = q/tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.math.pow(q,2)+1e-10 ,axis=-1)), axis=-1)
    

        n = tf.math.multiply_no_nan(k,q)
        n = tf.transpose(n, perm=self.p) 
        v = tf.math.multiply_no_nan(n,v)
        v = tf.reduce_sum(v, axis=-1)

        return v
##================================================= Model Architecture
inputs = tf.keras.Input(shape=(18))
x = inputs
x = Symmetry_Set_Basis(num_out=1, rank=tf.rank(x))(x, x, x)
x = Operator_Basis(num_out=1,rank=tf.rank(x))(x, x, x)
x = Symmetry_Set_Basis(num_out=1, rank=tf.rank(x))(x, x, x)
x = Operator_Basis(num_out=1,rank=tf.rank(x))(x, x, x)
x = tf.keras.layers.Dense(1)(x)
modelANN = tf.keras.Model(inputs= inputs, outputs=x, name='ANN')
        
#---------------------------------------------------------- Call Backs
model_type = "ANN"
save_dir = './test1/'
model_name = '%s_model_'% model_type 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

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

modelANN.fit(np.log(np.abs(x_rg)+1), np.log(np.abs(y_rg)+1), callbacks = callbacks, shuffle=True , epochs=30, batch_size=32, verbose=1) #you need to set validation data


y_pre = modelANN.predict(np.log(np.abs(x_rg)+1))
plt.title("Analytic NN - Regression(input(,18))")
plt.xlabel("Model Prediction")
plt.ylabel("Target")
plt.scatter(y_pre,np.log(np.abs(y_rg)+1))
plt.savefig("./plot/Regression_Scatter.png")
plt.show()

plt.title("Analytic NN - Regression(input(,18))")
plt.xlabel("Target space")
plt.ylabel("Number of data")
plt.hist(y_pre, histtype='step', label = "model prediction")
plt.hist(np.log(np.abs(y_rg)+1), histtype='step',  label = "target")
plt.savefig("./plot/Regression_Hist.png")
plt.legend()
plt.show()

modelANN.save("./pre_train_models")



        
        
