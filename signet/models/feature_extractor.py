import tensorflow as tf
from keras.layers.convolutional.base_depthwise_conv import DepthwiseConv

class DepthwiseConv1D(DepthwiseConv):

    def __init__(
        self,
        kernel_size,
        strides=1,
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def call(self, inputs):
        if self.data_format == "channels_last":
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)
        dilation_rate = (1,) + self.dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            dilations=dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format, ndim=4
            ),
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(
                    self.data_format, ndim=4
                ),
            )

        outputs = tf.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            input_dim = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            input_dim = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        input_dim = conv_utils.conv_output_length(
            input_dim,
            self.kernel_size[0],
            self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, input_dim)
        elif self.data_format == "channels_last":
            return (input_shape[0], input_dim, out_filters)

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x
    
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply


def CNN1DFeatureExtractor(CFG):
    inp = tf.keras.Input((CFG.max_len,CFG.CHANNELS))
    x = tf.keras.layers.Masking(mask_value=CFG.PAD[0],input_shape=(CFG.max_len,CFG.CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(CFG.dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(CFG.dim,expand=2)(x)

    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(CFG.dim,expand=2)(x)

    if CFG.dim == 384: #for the 4x sized model
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(CFG.dim,expand=2)(x)

        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(CFG.dim,expand=2)(x)

    x = tf.keras.layers.Dense(CFG.dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=CFG.dropout_step)(x)
    x = tf.keras.layers.Dense(CFG.NUM_CLASSES,name='classifier')(x)
    return tf.keras.Model(inp, x)