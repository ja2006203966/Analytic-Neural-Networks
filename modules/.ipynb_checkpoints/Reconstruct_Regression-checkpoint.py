import numpy as np
import tensorflow as tf 
from Reconstruct import Symmetry_Set_Basis_Reconstruct, Operator_Basis_Reconstruct, Selection_Reconstruct, rgsn_Reconstruct, Out_Analytic_Set

def Regression_Reconstruct(model_path, x_rg, y_rg):
    test = tf.keras.models.load_model(model_path)
    a = np.empty([18], dtype=np.str)
    x, y = tf.constant(x_rg), tf.constant(y_rg)
    x, n = Symmetry_Set_Basis_Reconstruct(num_out=1, rank=tf.rank(x), weights =test.layers[1].get_weights())(x, x, x)
    n1, ind1 = Selection_Reconstruct(n)
    a = Out_Analytic_Set(a, n1, ind1, rgsnw=1, rgsnb=0, mode = "SSB")
    x, n, rgsn = Operator_Basis_Reconstruct(num_out=1,rank=tf.rank(x), weights =test.layers[2].get_weights())(x, x, x)
    n2, ind2 = Selection_Reconstruct(n, ind=ind1)
    rgsnw1, rgsnb1 = rgsn_Reconstruct(rgsn)
    a = Out_Analytic_Set(a, n2, ind2, rgsnw=rgsnw1, rgsnb=rgsnb1, mode = "OB")
    
    x, n = Symmetry_Set_Basis_Reconstruct(num_out=1, rank=tf.rank(x), weights =test.layers[3].get_weights())(x, x, x)
    n1, ind1 = Selection_Reconstruct(n, ind=ind2)
    a = Out_Analytic_Set(a, n1, ind1, rgsnw=1, rgsnb=0, mode = "SSB")
    x, n, rgsn = Operator_Basis_Reconstruct(num_out=1,rank=tf.rank(x), weights =test.layers[4].get_weights())(x, x, x)
    n2, ind2 = Selection_Reconstruct(n, ind=ind1)
    rgsnw1, rgsnb1 = rgsn_Reconstruct(rgsn)
    a = Out_Analytic_Set(a, n2, ind2, rgsnw=rgsnw1, rgsnb=rgsnb1, mode = "OB")
    
    x = tf.keras.layers.Dense(1, weights =test.layers[5].get_weights())(x)
    return x, a


