import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import sfm
import pdb

class SFMSeq(nn.Module):
    def __init__(self, input_size, hidden_size, freq_size, output_size, seq_len):
        super(SFMSeq, self).__init__()
        self.sfm_cell = sfm.SFM(input_size, hidden_size, freq_size, output_size)
        self.seq_len = seq_len
        self.output_size = output_size
        self.fc = nn.Linear(output_size,output_size)
        
    def forward(self, input_seq):
        # pdb.set_trace()
        # Initialize states
        batch_size = input_seq.shape[0]
        h, c, re_s, im_s, time = self.sfm_cell.init_state()
    
        outputs = []
        for row in range(input_seq.shape[0]):
            batch_outputs = []
            for t in range(input_seq.shape[1]):
                out, h, c, re_s, im_s, time = self.sfm_cell(input_seq[row, t, :], h, c, re_s, im_s, time)
                batch_outputs.append(out)
    
                if (t + 1) % 10 == 0:
                    out_batch = torch.stack(batch_outputs, dim=1)
                    out_batch = self.fc(out_batch.T)
                    outputs.append(out_batch)
                    batch_outputs = []
    
            if len(batch_outputs) > 0:
                out_batch = torch.stack(batch_outputs, dim=1)
                out_batch = self.fc(out_batch.T)
                outputs.append(out_batch)
    
        if len(outputs) > 0:
            out_batch = torch.cat(outputs, dim=0)
        else:
            out_batch = torch.tensor([])
        out_batch = out_batch.view(batch_size, -1, self.output_size)
    
        return torch.squeeze(out_batch, dim=-1)
    
