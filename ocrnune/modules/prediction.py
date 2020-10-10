import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int):
        super(AttentionCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_embeddings = num_embeddings
        
        self.i2h = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.h2h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.score = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        
    def forward(self, prev_hidden, batch_hidden, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_hidden_proj = self.i2h(batch_hidden)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_hidden_proj + prev_hidden_proj)) # batch_size x num_encoder_step * 1
        
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_hidden).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1) # batch_size x (num_channel + num_embedding)
        current_hidden = self.rnn(concat_context, prev_hidden)
        return current_hidden, alpha
    
    
class Attention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.generator = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def _char_to_one_hot(self, input_char: torch.Tensor, one_hot_dim: int = 38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, one_hot_dim).zero_()
        return one_hot
    
    def forward(self, batch_hidden, text, is_train: bool = True, batch_max_length: int = 25):
        """
        Args:
            batch_hidden ([type]): contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text ([type]): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
            is_train (bool, optional): [description]. Defaults to True.
            batch_max_length (int, optional): [description]. Defaults to 25.
             

        Returns:
            torch.Tensor : probability distribution at each step [batch_size x num_steps x num_classes]
        """
        
        batch_size = batch_hidden.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.
        
        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0))
        
        
        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_one_hot(text[:, i], one_hot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_hidden : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_hidden, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
            
        else:
            targets = torch.LongTensor(batch_size).fill_(0)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)
            
            for i in range(num_steps):
                char_onehots = self._char_to_one_hot(targets, one_hot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_hidden, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input
                
        return probs