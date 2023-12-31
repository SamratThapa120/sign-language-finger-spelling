import tensorflow as tf

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
    def __init__(self, dim=256, num_heads=4, dropout=0,attention_span=0,lookahead=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True
        self.attention_span=attention_span
        self.lookahead = lookahead

    def generate_mask(self,input_shape, attention_span, lookahead=True):
        seq_len = input_shape[1]  # assuming input_shape is (batch_size, seq_len, dim)
        idxs = tf.range(seq_len)
        if lookahead:
            mask = tf.abs(idxs[None, :] - idxs[:, None]) <= attention_span
        else:
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)==1
            mask = tf.logical_and(mask,tf.abs(idxs[None, :] - idxs[:, None]) <= attention_span)
        # add extra dimensions to match the shape of attn
        mask = mask[None, None, :, :]
        return tf.cast(mask, dtype=tf.float32)


    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]
        elif self.attention_span > 0:
            mask = self.generate_mask(tf.shape(inputs), self.attention_span,self.lookahead)

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(CFG,expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=CFG.dim,num_heads=CFG.num_heads,dropout=attn_dropout,attention_span=CFG.attention_span,lookahead=CFG.lookahead)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(CFG.dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(CFG.dim, use_bias=False)(x)
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
def conv_output_length(input_length, filter_size, padding, stride):
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride
def apply_times(input_length, filter_size, padding, stride,labellen=45):
    curlen = input_length
    i =0
    while curlen>labellen:
        curlen = conv_output_length(curlen, filter_size, padding, stride)
        i+=1
    return i

def Cnn1dMhsaFeatureExtractor(CFG):
    inp = tf.keras.Input((CFG.max_len, CFG.CHANNELS))
    x = inp

    # Create the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=CFG.max_len, output_dim=CFG.dim, input_length=CFG.max_len)
    embeddings = embedding_layer(tf.range(0, CFG.max_len))
    embeddings = tf.expand_dims(embeddings, axis=0)  # Expand dimensions to match batch size

    # x = tf.keras.layers.Masking(mask_value=CFG.PAD[0],input_shape=(CFG.max_len,CFG.CHANNELS))(inp)
    ksize = CFG.kernel_size
    x = tf.keras.layers.Dense(CFG.dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)
    if CFG.positional_encodings_once:
        x = tf.keras.layers.Add()([x, embeddings])
        x = tf.keras.layers.Dropout(CFG.positional_dropout)(x)

    downsample = CFG.num_feature_blocks - apply_times(CFG.max_len, CFG.kernel_size_downsampling, "valid", CFG.downsampling_strides, CFG.MAX_WORD_LENGTH)

    for i in range(CFG.num_feature_blocks):
        x = Conv1DBlock(CFG.dim, ksize, drop_rate=CFG.blocks_dropout)(x)
        x = Conv1DBlock(CFG.dim, ksize, drop_rate=CFG.blocks_dropout)(x)
        x = Conv1DBlock(CFG.dim, ksize, drop_rate=CFG.blocks_dropout)(x)

        # Add the embeddings to x before the transformer layer
        # if not CFG.positional_encodings_once:
        #     x = tf.keras.layers.Add()([x, embeddings])
        #     x = tf.keras.layers.Dropout(CFG.positional_dropout)(x)

        x = TransformerBlock(CFG, expand=2)(x)
        if i > downsample and CFG.do_downsample:
            x = tf.keras.layers.DepthwiseConv1D(
                (CFG.kernel_size_downsampling,),
                strides=CFG.downsampling_strides,
                dilation_rate=1,
                padding='valid',
                use_bias=False,
                depthwise_initializer='glorot_uniform',
                name=str(i) + 'downsample_conv')(x)
    x = tf.keras.layers.Dense(CFG.dim*2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.Dropout(CFG.final_dropout)(x)
    x = tf.keras.layers.Dense(CFG.NUM_CLASSES, name='classifier')(x)
    return tf.keras.Model(inp, x)


