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
        # x = self.causal_pad(inputs)
        x = self.dw_conv(inputs)
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

#https://www.kaggle.com/code/markwijkhuizen/aslfr-transformer-training-inference?scriptVersionId=135498607&cellId=46


class DecoderMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model, num_of_heads, dropout, d_out=None):
        super(DecoderMultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model//num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth//2, use_bias=False) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out, use_bias=False)
        self.softmax = tf.keras.layers.Softmax()
        self.do = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True
        self.scale = self.d_model ** -0.5

    def call(self, q, k, v, attention_mask=None, training=False):
        
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](q)
            K = self.wk[i](k)
            V = self.wv[i](v)
            attn = tf.matmul(Q, K, transpose_b=True) * self.scale
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]

            attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=attention_mask)

            x = attn @ V
            multi_attn.append(x)
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        multi_head_attention = self.do(multi_head_attention, training=training)
        
        return multi_head_attention


def TransformerBlock(CFG,expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=CFG.dim,num_heads=CFG.num_heads,dropout=attn_dropout)(x)
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


# Decoder based on multiple transformer blocks
class Decoder(tf.keras.layers.Layer):
    def __init__(self, CFG,attn_dropout=0.2):
        super(Decoder, self).__init__(name='decoder')
        self.num_blocks = CFG.decoder_blocks
        self.supports_masking = True
        self.max_len = CFG.MAX_WORD_LENGTH
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros([1,CFG.MAX_WORD_LENGTH, CFG.dim]),
            trainable=True,
            name='embedding_positional_encoder',
        )
        # Character Embedding
        # self.char_emb = tf.keras.layers.Embedding(CFG.NUM_CLASSES, CFG.dim, embeddings_initializer=tf.keras.initializers.constant(0.0))
        # Positional Encoder MHA
        # self.pos_emb_mha = MultiHeadSelfAttention(dim=CFG.dim,num_heads=CFG.num_heads,dropout=attn_dropout)
        # self.pos_emb_ln = tf.keras.layers.LayerNormalization(epsilon=CFG.layer_norm_eps)
        # First Layer Normalisation
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=CFG.layer_norm_eps))
            # Multi Head Attention
            self.mhas.append(DecoderMultiHeadAttention(d_model=CFG.dim,num_of_heads=CFG.num_heads,dropout=attn_dropout))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=CFG.layer_norm_eps))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(CFG.dim * CFG.transformer_mlp_expand_ratio, activation=tf.keras.activations.gelu, kernel_initializer=tf.keras.initializers.glorot_uniform, use_bias=False),
                tf.keras.layers.Dropout(CFG.decoder_mlp_dropout),
                tf.keras.layers.Dense(CFG.dim, kernel_initializer=tf.keras.initializers.he_uniform, use_bias=False),
            ]))
            
    def get_causal_attention_mask(self, B):
        i = tf.range(self.max_len)[:, tf.newaxis]
        j = tf.range(self.max_len)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (self.max_len, self.max_len))
        # mult = tf.concat(
        #     [tf.expand_dims(B, -1), tf.constant([1, 1], dtype=tf.int32)],
        #     axis=0,
        # )
        # mask = tf.tile(mask, mult)
        mask = tf.cast(mask, tf.float16)
        return mask
        
    def call(self, encoder_outputs):
        # Batch Size
        B = tf.shape(encoder_outputs)[0]
        # Cast to INT32
        # phrase = tf.cast(phrase, tf.int32)
        # Prepend SOS Token
        # phrase = tf.pad(phrase, [[0,0], [1,0]], constant_values=SOS_TOKEN, name='prepend_sos_token')
        # Pad With PAD Token
        # phrase = tf.pad(phrase, [[0,0], [0,self.max_len-MAX_PHRASE_LENGTH-1]], constant_values=PAD_TOKEN, name='append_pad_token')
        # Causal Mask
        # causal_mask = self.get_causal_attention_mask(B)
        # Positional Embedding
        x = tf.tile(self.positional_embedding, [B, 1, 1]) # + self.char_emb(phrase)
        # Causal Attention
        # x = self.pos_emb_ln(x + self.pos_emb_mha(x, x, x, attention_mask=causal_mask))
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(tf.add(x, mha(x, encoder_outputs, encoder_outputs)))
            x = ln_2(tf.add(x,mlp(x)))
        # Slice 31 Characters
        # x = tf.slice(x, [0, 0, 0], [-1, MAX_PHRASE_LENGTH, -1])
        return x 

def EncoderDecoder(CFG):
    inp = tf.keras.Input((CFG.max_len,CFG.CHANNELS))
    x = inp
    # x = tf.keras.layers.Masking(mask_value=CFG.PAD[0],input_shape=(CFG.max_len,CFG.CHANNELS))(inp)
    ksize = CFG.kernel_size
    x = tf.keras.layers.Dense(CFG.dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    for _ in range(CFG.num_feature_blocks):
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=CFG.blocks_dropout)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=CFG.blocks_dropout)(x)
        x = Conv1DBlock(CFG.dim,ksize,drop_rate=CFG.blocks_dropout)(x)
        x = TransformerBlock(CFG,expand=CFG.transformer_mlp_expand_ratio)(x)
    
    x = Decoder(CFG)(x)
    x = tf.keras.layers.Dense(CFG.dim*2,activation=None,name='top_conv')(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = LateDropout(0.2, start_step=dropout_step)(x)
    x = tf.keras.layers.Dropout(CFG.final_dropout)(x)
    x = tf.keras.layers.Dense(CFG.NUM_CLASSES,name='classifier')(x)

    
    return tf.keras.Model(inp, x)
