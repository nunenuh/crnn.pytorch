import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.generator = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def _char_to_onehot(self, input_char: torch.Tensor, onehot_dim: int = 38):
        used_device = input_char.get_device()
        if used_device == -1: used_device = 'cpu'
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(used_device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot
    
    def forward(self, feature, text=None, max_length: int = 25):
        """
        Args:
            feature ([type]): contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text ([type]): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
            is_train (bool, optional): [description]. Defaults to True.
            batch_max_length (int, optional): [description]. Defaults to 25.
             

        Returns:
            torch.Tensor : probability distribution at each step [batch_size x num_steps x num_classes]
        """
        used_device = feature.get_device()
        if used_device == -1: used_device = 'cpu'
        
        batch_size = feature.size(0)
        hidden_size = self.hidden_size
        num_class = self.num_classes
        num_steps = max_length + 1  # +1 for [s] at end of sentence.
        
        hidden = (torch.FloatTensor(batch_size, hidden_size).fill_(0).to(used_device),
                  torch.FloatTensor(batch_size, hidden_size).fill_(0).to(used_device))
        
        
        if text!=None: # training data is used
            output_hiddens = torch.FloatTensor(batch_size, num_steps, hidden_size).fill_(0).to(used_device)
            
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=num_class).to(used_device)
                
                # hidden : decoder's hidden s_{t-1}, feature : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, feature, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
            
        else:
            targets = torch.LongTensor(batch_size).fill_(0)
            probs = torch.FloatTensor(batch_size, num_steps, num_class).fill_(0)
            
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=num_class).to(used_device)
                
                hidden, alpha = self.attention_cell(hidden, feature, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input
                
        return probs
    
    
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
        # self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        
        
    def forward(self, prev_hidden, feature, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        feature_proj = self.i2h(feature)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        
        e = self.score(torch.tanh(feature_proj + prev_hidden_proj)) # batch_size x num_encoder_step * 1
        
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), feature).squeeze(1)  # batch_size x num_channel
#         print(f'context.shape: {context.shape}' )
#         print(f'char_onehots.shape: {char_onehots.shape}' )
        
        concat_context = torch.cat([context, char_onehots], dim=1) # batch_size x (num_channel + num_embedding)
        current_hidden = self.rnn(concat_context, prev_hidden)
        return current_hidden, alpha
    