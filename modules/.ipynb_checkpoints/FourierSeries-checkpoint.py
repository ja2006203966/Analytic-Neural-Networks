import tensorflow as tf
import numpy as np
class FourierSeries(tf.keras.layers.Layer):
    def __init__(self, rank=2, node = 10):
        super(FourierSeries, self).__init__()
        self.p = [[0,2,1],[0,1,3,2], [0,1,2,4,3], [0,1,2,3,5,4]][rank-2]
        self.node = node
    def call(self, cn):
        m = self.node
        n = tf.cast(tf.rank(cn), tf.int32)
        one = tf.constant(1, dtype=tf.float32)
        tileshape = tf.concat( [tf.cast(tf.linspace(one, one, n), tf.int32), tf.constant([m]) ], axis=0)
        order = tf.linspace(one, m, m)
        order = tf.reshape(order, tileshape)
        vp = tf.expand_dims(cn, axis=-1)
        vp = tf.tile(vp, tileshape)
        order = tf.tile(order, tf.concat([tf.shape(vp)[:-1], [1]], axis=-1))
        order = tf.cast(order, tf.float32)
#         order = order/self.step
        vsin = tf.math.sin(np.pi*2*order*vp)
        vcos = tf.math.cos(np.pi*2*order*vp)
        v0 = tf.reduce_min(vp*0, axis=-1)
        v0 = tf.expand_dims(v0, axis=-1)
        vp = tf.concat([v0, vsin, vcos], axis=-1)
        return vp