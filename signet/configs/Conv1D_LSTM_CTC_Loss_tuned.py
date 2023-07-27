from .base import Base
class Conv1D_LSTM_CTC_Loss(Base):
    
    #Strategy params
    device="GPU"
    num_devices=1

    #feature extractor params
    kernel_size=12
    #dataset params
    char_to_idx = {" ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"\/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    merge_repeated=True
    max_len=384
    drop_remainder=False
    augment=True
    flip_lr_probability=0.5
    random_affine_probability=0.2
    freeze_probability=0.1

    #training params
    validation_prediction_save_ratio=0.1
    is_jit=True
    summary=True
    one_hot=False
    save_output = True
    output_dir = '../runs/ctcloss_with_frozenframes'
    
    seed = 42
    verbose = 1 #0) silent 1) progress bar 2) one line per epoch
    
    num_feature_blocks=5
    blocks_dropout=0.3
    replicas = num_devices
    lr = 0.001370400500342758
    weight_decay = 0.00038476304486437685
    lr_min = 1.2417933819232701e-05
    epoch = 100 
    train_epochs = epoch
    warmup = 0
    batch_size = 128
    val_batch_size = 128
    validation_frequency=2
    snapshot_epochs = []
    swa_epochs = [] #list(range(epoch//2,epoch+1))
    
    fp16 = True
    decay_type = 'cosine'
    dim = 2**7
 
    blank_index=len(char_to_idx)
    ctc_decoder = "greedy"

    start_epoch=0  
    resume = 0
    resume_path=None 
    save_frequency=5