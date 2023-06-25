from .base import Base
class Conv1D_LSTM_CTC_Loss(Base):
    #Strategy params
    device="GPU"


    #feature extractor params
    dropout_step=0
    dim=192
    feature_dim=512

    #dataset params
    batch_size=64
    max_len=64
    drop_remainder=False
    augment=False
    shuffle=False
    repeat=False