from .base import Base
import tensorflow as tf
class ctc_loss_encdec_params(Base):
    
    POINT_LANDMARKS = Base.LIP + Base.LHAND + Base.RHAND+ Base.POSE
    ERASABLE_LANDMARKS = Base.LIP + Base.POSE
    NUM_NODES = len(POINT_LANDMARKS)
    CHANNELS = 6*NUM_NODES

    
    MAX_WORD_LENGTH=45

    #Strategy params
    device="GPU"
    num_devices=1

    #feature extractor params
    kernel_size=12

    #dataset params
    num_parallel_reads = tf.data.AUTOTUNE
    char_to_idx = {" ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    merge_repeated=True
    max_len=384

    drop_remainder=False
    augment=True
    flip_lr_probability=0.5
    random_affine_probability=0.8
    freeze_probability=0.1
    tempmask_probability=0.8
    tempmask_range = (0.1,0.4)
    erase_probability=0.8
    #training params
    validation_prediction_save_ratio=0.1
    is_jit=True
    summary=True
    one_hot=False
    save_output = True
    output_dir = '../runs/top_tuned_models'
    
    seed = 42
    verbose = 1 #0) silent 1) progress bar 2) one line per epoch
    
    num_feature_blocks=7
    blocks_dropout=0.1

    replicas = num_devices
    lr = 0.001
    weight_decay = 0.003691291783715857
    lr_min = 1.5e-06

    epoch = 100 
    train_epochs = 100
    warmup_epochs = 20
    batch_size = 128
    val_batch_size = 128
    validation_frequency=1
    snapshot_epochs = []
    swa_epochs = [] #list(range(epoch//2,epoch+1))
    
    fp16 = True
    decay_type = 'cosine'
    warmup_type = "linear"
    dim = 2**8
 
    # start_index=len(char_to_idx)
    blank_index=len(char_to_idx)
    pad_index=len(char_to_idx)+1
    loss_pad_index=pad_index

    NUM_CLASSES0 = len(idx_to_char)
    # idx_to_char[start_index]=""
    idx_to_char[blank_index]="凸"

    NUM_CLASSES = NUM_CLASSES0+1

    start_epoch=0 
    resume_path=None
    save_frequency=5
    num_heads=8
    layer_norm_eps=1e-6
    decoder_mlp_dropout = 0.2
    transformer_mlp_expand_ratio=2
    decoder_blocks=1
    final_dropout=0.2

    use_mask=False
    attention_span=0
    kernel_size_downsampling=12
    downsampling_strides=2
    do_downsample=True

    loss_type="focal"    #focal or min_wer
    #focal error params
    alpha=0.5
    gamma=0.5

    #min_wer params
    beam_width=8

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))}
