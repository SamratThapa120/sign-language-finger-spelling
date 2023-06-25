from .base import Base
class Conv1D_LSTM_CTC_Loss(Base):
    device="GPU"
    batch_size=64
    max_len=64
    drop_remainder=False
    augment=False
    shuffle=False
    repeat=False
    feature_dim=512