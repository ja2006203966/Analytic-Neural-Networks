import tensorflow as tf

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