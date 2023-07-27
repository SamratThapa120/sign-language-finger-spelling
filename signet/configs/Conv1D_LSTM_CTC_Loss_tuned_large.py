from .base import Base
import tensorflow as tf
class Conv1D_LSTM_CTC_Loss(Base):
    
    #Strategy params
    device="GPU"
    num_devices=1

    #feature extractor params
    kernel_size=11

    #dataset params
    num_parallel_reads = tf.data.AUTOTUNE
    char_to_idx = {" ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    merge_repeated=True
    max_len=384
    drop_remainder=False
    augment=True
    flip_lr_probability=0.3
    random_affine_probability=0.3
    freeze_probability=0.3
    tempmask_probability=0.3
    #training params
    validation_prediction_save_ratio=0.1
    is_jit=True
    summary=True
    one_hot=False
    save_output = True
    output_dir = '../runs/ctc_with_frozenframes_XL'
    
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
 
    blank_index=len(char_to_idx)
    ctc_decoder = "greedy"

    start_epoch=0 
    resume_path=None
    save_frequency=5
    num_heads=8
        
    @classmethod
    def to_dict(cls):
        return {attr: getattr(cls, attr) for attr in dir(cls) if not attr.startswith('__') and not callable(getattr(cls, attr))}