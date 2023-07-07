from .base import Base
import torch
class Conv1D_LSTM_CTC_Loss(Base):
    
    #Strategy params
    num_devices=torch.cuda.device_count()
    device="cuda" if num_devices>0 else "cpu"
    print("num_devices:",num_devices)

    #feature extractor params
    dropout_step=0
    dim=192
    feature_dim=512

    #dataset params
    blank_index=0
    char_to_idx = {"|":-1," ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
    char_to_idx = {k:v+1 for k,v in char_to_idx.items()}
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    merge_repeated=True
    max_len=384
    drop_remainder=False
    augment=False
    shuffle=False

    #training params
    beta1=0.9
    rho=0.95
    eps=0.00000001
    grad_clip=5
    zero_infinity=True
    validation_prediction_save_ratio=0.1
    is_jit=True
    summary=True
    one_hot=False
    n_splits = 5
    save_output = True
    output_dir = '../runs/conv1d_mhsa_ctcloss'
    
    seed = 42
    verbose = 1 #0) silent 1) progress bar 2) one line per epoch
    
    replicas = num_devices
    lr = 5e-4* replicas
    weight_decay = 0.000001
    lr_min = 1e-6
    epoch = 300 
    warmup = 0
    batch_size = 512 * replicas
    num_workers_train=8
    num_workers_valid=1
    val_batch_size = 1
    validation_freq=5

    snapshot_epochs = []
    swa_epochs = [] #list(range(epoch//2,epoch+1))
    
    fp16 = True
    fgm = False
    awp = True
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    decay_type = 'cosine'
    dim = 192
 
    ctc_decoder = "greedy"

    start_epoch=0  
    resume = 0
    resume_path=None 
    save_frequency=5