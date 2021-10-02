import torch.nn as nn


# class Encoder(nn.Module):

#   def __init__(self,batch_size, seq_len, n_features, embedding_dim=64):
#     super(Encoder, self).__init__()

#     self.batch_size = batch_size
#     self.seq_len, self.n_features = seq_len, n_features
#     self.embedding_dim, self.hidden_dim = int(embedding_dim), int(embedding_dim)

#     self.rnn1 = nn.LSTM(
#       input_size=self.n_features,
#       hidden_size=self.hidden_dim,
#       num_layers=1,
#       batch_first=True
#     )
    
#     # self.rnn2 = nn.LSTM(
#     #   input_size=self.hidden_dim,
#     #   hidden_size=embedding_dim,
#     #   num_layers=1,
#     #   batch_first=True
#     # )

#   def forward(self, x):
#     x = x.reshape((x.shape[0], self.seq_len, self.n_features))
#     x, (hidden_n, _) = self.rnn1(x.float())
#     # x, (hidden_n, _) = self.rnn2(x)
#     return hidden_n.reshape((x.shape[0], self.embedding_dim))



# class Decoder(nn.Module):

#   def __init__(self, batch_size, seq_len, input_dim=64, n_features=1):
#     super(Decoder, self).__init__()
#     self.batch_size = batch_size

#     self.seq_len, self.input_dim = seq_len, input_dim
#     self.hidden_dim, self.n_features = input_dim, n_features
#     self.hidden_dim=64
#     self.rnn1 = nn.LSTM(
#       input_size=input_dim,
#       hidden_size=self.hidden_dim,
#       num_layers=2,
#       batch_first=True,
#       dropout = 0.2
#     )

#     # self.rnn2 = nn.LSTM(
#     #   input_size=input_dim,
#     #   hidden_size=self.hidden_dim,
#     #   num_layers=1,
#     #   batch_first=True,
#     #   dropout = 0.2
#     # )

#     self.output_layer = nn.Linear(self.hidden_dim, n_features)

#   def forward(self, x):
#     b_size = x.shape[0]
#     x = x.repeat(self.seq_len, self.n_features)
#     x = x.reshape((-1, self.seq_len, self.input_dim))

#     x, (hidden_n, cell_n) = self.rnn1(x)
#     # x, (hidden_n, cell_n) = self.rnn2(x)
#     x = x.reshape((b_size,self.seq_len, self.hidden_dim))
#     x = self.output_layer(x)
#     return x

class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        x, (hidden_state, cell_state) = self.LSTM1(x.reshape(-1,self.seq_len,self.no_features))  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 2,
            dropout = 0.2,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class RecurrentAutoencoder(nn.Module):

  def __init__(self, batch_size,seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).cuda()
    self.decoder = Decoder(seq_len, 20, n_features).cuda()

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x
