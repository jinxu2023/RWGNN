import tensorflow as tf
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.regularizers import l2


class GCN_Layer(Layer):
    def __init__(self, units=200, use_bias=False, activation=tf.nn.relu, l2_reg=0, seed=0, **kwargs):
        super(GCN_Layer, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1][-1], self.units),
                                 initializer=GlorotUniform(self.seed),
                                 regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        A, H = inputs
        A = A + tf.eye(A.shape[0])
        row_sum = tf.reduce_sum(A, axis=1, keepdims=True)
        row_sum = row_sum + (row_sum == 0).astype('float32')
        row_norm = row_sum ** -0.5
        A_norm = row_norm * A * row_norm.T
        if self.activation is not None:
            return self.activation(tf.einsum('ij,jk,kl->il', A_norm, H, self.W))
        else:
            return tf.einsum('ij,jk,kl->il', A_norm, H, self.W)


class SGC_Layer(GCN_Layer):
    def __init__(self, units=200, use_bias=False, seed=0, **kwargs):
        super(SGC_Layer, self).__init__(units=units, use_bias=use_bias, seed=seed, activation=None)


class GraphSage_Layer(Layer):
    def __init__(self, units=200, activation=tf.nn.relu, l2_reg=0,
                 normalize=True, seed=0, **kwargs):
        super(GraphSage_Layer, self).__init__()
        self.units = units
        self.activation = activation
        self.l2_reg = l2_reg
        self.normalize = normalize
        self.seed = seed

    def build(self, input_shapes):
        input_dim = input_shapes[-1][-1]
        self.W = self.add_weight(name='W',
                                 shape=(input_dim, self.units),
                                 initializer=GlorotUniform(self.seed),
                                 regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        A, H = inputs
        H = self.activation(tf.einsum('ij,jk,kl->il', A, H, self.W))
        if self.normalize:
            H = tf.nn.l2_normalize(H, axis=1)
        return H


class GAT_Layer(Layer):
    def __init__(self, attention_num=8, units=8,
                 activation=(tf.nn.leaky_relu, tf.nn.elu), l2_reg=0.0005,
                 is_output_layer=False, seed=1024, **kwargs):
        super(GAT_Layer, self).__init__()
        self.attention_num = attention_num
        self.units = units
        self.activation = activation
        self.l2_reg = l2_reg
        self.is_output_layer = is_output_layer
        self.seed = seed

    def build(self, input_shapes):
        self.W = [self.add_weight(name='W_' + str(i),
                                  shape=(input_shapes[-1][-1], self.units),
                                  initializer=GlorotUniform(self.seed),
                                  regularizer=l2(self.l2_reg))
                  for i in range(self.attention_num)]  # K*[F*F']
        self.a = [self.add_weight(name='a_' + str(i),
                                  shape=(2 * self.units, 1),
                                  initializer=GlorotUniform(self.seed),
                                  regularizer=l2(self.l2_reg))
                  for i in range(self.attention_num)]  # K*[(2*F')*1]
        self.a_left = [self.add_weight(name='a_left_' + str(i),
                                       shape=(self.units,),
                                       initializer=GlorotUniform(self.seed),
                                       regularizer=l2(self.l2_reg))
                       for i in range(self.attention_num)]
        self.a_right = [self.add_weight(name='a_right_' + str(i),
                                        shape=(self.units,),
                                        initializer=GlorotUniform(self.seed),
                                        regularizer=l2(self.l2_reg))
                        for i in range(self.attention_num)]

    def call(self, inputs, **kwargs):
        A, H = inputs  # N*F
        node_num = H.shape[0]
        gather_index = tf.where(tf.ones((node_num, node_num)))  # (N*N)*2

        Alpha = [None] * self.attention_num
        for i in range(self.attention_num):
            e_left = tf.einsum('ij,jk,k->i', H, self.W[i], self.a_left[i])
            e_right = tf.einsum('ij,jk,k->i', H, self.W[i], self.a_right[i])
            e_stacked = tf.stack([e_left, e_right], axis=1)
            e_gathered = tf.gather(e_stacked, gather_index, axis=0)
            e_gathered = e_gathered[:, 0, 0] + e_gathered[:, 1, 1]
            E = tf.reshape(e_gathered, shape=(node_num, node_num))
            Alpha[i] = tf.math.exp(E) / tf.reduce_sum(tf.math.exp(E * A), axis=1, keepdims=True)

        H_out = [None] * self.attention_num
        for i in range(self.attention_num):
            Alpha[i] = Alpha[i] * A
            H_out[i] = self.activation[1](tf.matmul(tf.matmul(Alpha[i], H), self.W[i]))  # N*F'
        if self.is_output_layer:
            H_out = tf.reduce_mean(tf.stack(H_out, axis=-1), axis=-1)  # N*F'
        else:
            H_out = tf.concat(H_out, axis=1)  # N*(F'*K)
        return H_out

    def compute_output_shape(self, input_shape):
        if self.is_output_layer:
            return input_shape[0], self.units
        else:
            return input_shape[0], self.units * self.attention_num


class LightGCN_Layer(Layer):
    def __init__(self, units=200, seed=0, **kwargs):
        super(LightGCN_Layer, self).__init__()
        self.units = units
        self.seed = seed

    def call(self, inputs, **kwargs):
        A, H = inputs
        row_sum = tf.reduce_sum(A, axis=1, keepdims=True)
        row_sum = row_sum + (row_sum == 0).astype('float32')
        row_norm = row_sum ** -0.5
        A_norm = row_norm * A * row_norm.T
        return A_norm @ H


class GIN_Layer(Layer):
    def __init__(self, units=200, use_bias=True, activation=tf.nn.relu, l2_reg=0, seed=0, **kwargs):
        super(GIN_Layer, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        self.e = self.add_weight(name='e',
                                 shape=(),
                                 initializer=GlorotUniform(self.seed))
        self.MLP = Dense(units=self.units,
                         activation=self.activation,
                         use_bias=self.use_bias,
                         kernel_initializer=GlorotUniform(self.seed),
                         kernel_regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        A, H = inputs
        row_sum = tf.reduce_sum(A, axis=1, keepdims=True)
        row_sum = row_sum + (row_sum == 0).astype('float32')
        row_norm = row_sum ** -0.5
        A_norm = row_norm * A * row_norm.T
        H_out = self.MLP((1 + self.e) * H + A_norm @ H)
        return H_out
