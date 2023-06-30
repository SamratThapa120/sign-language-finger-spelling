import torch
from typing import List


#https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#greedy-decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        
    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices.numpy() if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined
    
class CTCLossBatchFirst(torch.nn.Module):
    def __init__(self,blank=0,zero_infinity=False):
        super().__init__()
        self.lossf = torch.nn.CTCLoss(blank=blank,zero_infinity=zero_infinity)
        
    def forward(self, input_features, labels, inp_length, target_length):
        return self.lossf(torch.transpose(input_features,0,1),labels,inp_length,target_length)