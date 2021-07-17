import tensorflow as tf
import numpy as np
## This module is used to reconstruct Analytics function
class Symmetry_Set_Basis_Reconstruct(tf.keras.layers.Layer):
    def __init__(self, node=7, num_out=1, rank=2):
        super(Symmetry_Set_Basis_Reconstruct, self).__init__()
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
##--------------------------------------------------------
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
##-----------------------------------------------------------------------------------------
        n = tf.math.multiply_no_nan(k,q)
##--------------------------------------------------sum all v*n version ------------------------------------------
        n = tf.transpose(n, perm=self.p)
        v = tf.math.multiply_no_nan(n,v)
        v = tf.reduce_sum(v, axis=-1)

        return v, n
        
class Operator_Basis_Reconstruct(tf.keras.layers.Layer):
    def __init__(self, node=3, num_out=1, rank=2):
        super(Operator_Basis_Reconstruct, self).__init__()
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
##--------------------------------------------------sum all v*n version ------------------------------------------
        v = tf.math.multiply_no_nan(n,v)
        v = tf.reduce_sum(v, axis=-1)

##----------------------------------------------------------------------------------------------
        return v, n, self.alpha.weights
        

def Selection_Reconstruct(n, r = 0.01, ind=[False]):
    n = tf.reduce_mean(n,axis=0)
#     n = tf.reduce_sum(n, axis=0)
    sl1 = tf.reduce_max(tf.math.abs(n), axis = -1)
    sl1 = sl1.numpy()
    node = len(sl1)
    dl = np.max(np.abs(sl1))*r
    index = np.linspace(0, node-1, node)[np.abs(sl1)>dl] 

    index = (set(ind))&(set(index)) if np.any(ind) else index

    index = np.array(list(index))
    index = index.astype(np.int32)
    n = n.numpy()
    n = np.take(n, index, axis=0)
#     n = np.sum(n, axis = 0)
    return n, index
def rgsn_Reconstruct(rgsn):
    return tf.squeeze(rgsn[0]).numpy(), tf.squeeze(rgsn[1]).numpy() # weight, bias

def Out_Analytic_Set(x, n, index, rgsnw=1, rgsnb=0, mode = "SSB"):
    n = np.around(n,4)
    n = n.astype(np.str)
    SSB_keys = ["vc1", "vc2", "vc3", "vc4", "vp2", "vp3", "vp4"]
    OB_keys = ["sqrt", "ln", "rgsn"]
    x0 = np.empty(x.shape, dtype=np.str) 
    x0 = x0.astype(np.dtype('<U32'))
    for i in range(len(x)):
        if i in index:
            print(i)
            
            w = np.take(n, np.linspace(0, len(index)-1, len(index))[index==i].astype(np.int32), axis=0)
            w = w.astype(np.str)
            w = np.squeeze(w)
            print(w)
            if mode == "SSB":
                for j in range(len(SSB_keys)):
                    x0[i] =x0[i] + w[j]+"*"+SSB_keys[j]+"("+x[i]+")"
            if mode == "OB":
                for j in range(len(OB_keys)):
                    x0[i] = x0[i] + w[j]+"*"+OB_keys[j]+"("+x[i]+")"
    return x0
                    
                
        


        
        
