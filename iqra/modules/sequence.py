import torch
import torch.nn as nn
# from typing import * 


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size:int, batch_first=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        
        self.rnn: nn.LSTM = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=batch_first)
        self.linear: nn.Linear = nn.Linear(hidden_size * 2, output_size)
        
        
    def forward(self, x: torch.Tensor):
        """[summary]
        
        x: visual features [batch x T x input_size]
        return x: contextual feature [batch x T x output_size]

        Args: 
            x ([torch.Tensor]): [visual features [batch x T x input_size]]

        Returns:
            [type]: [description]
        """
        
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x) # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent) # batch_size x T x output_size
        
        return output
        