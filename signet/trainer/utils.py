import torch
from typing import List
import logging


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
        original = "".join([self.labels[i.item()] for i in indices])
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices.numpy() if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined,original
    
class CTCLossBatchFirst(torch.nn.Module):
    def __init__(self,blank=0,zero_infinity=False):
        super().__init__()
        self.lossf = torch.nn.CTCLoss(blank=blank,zero_infinity=zero_infinity)
        
    def forward(self, input_features, labels, inp_length, target_length):
        return self.lossf(torch.transpose(input_features,0,1),labels,inp_length,target_length)
  
def get_logger(path):
    # Create a logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.DEBUG) # Set the logging level

    # Create a file handler
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG) # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # Set the logging level

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file and console handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
