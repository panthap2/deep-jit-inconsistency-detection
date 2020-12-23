import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        self.__rnn = nn.GRU(input_size=embedding_size,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=bidirectional)
    
    def forward(self, src_embedded_tokens, src_lengths, device):
        encoder_hidden_states, _ = self.__rnn.forward(src_embedded_tokens)
        encoder_final_state = encoder_hidden_states[torch.arange(
            src_embedded_tokens.size()[0], dtype=torch.int64, device=device), src_lengths-1]
        # encoder_final_state, _ = torch.max(encoder_hidden_states, dim=1)
        return encoder_hidden_states, encoder_final_state