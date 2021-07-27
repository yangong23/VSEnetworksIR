# ----------------------------------------------------------------
# Modified by Yan Gong
# Last revised: July 2021
# Reference: The orignal code is from VSRN: Visual Semantic Reasoning for Image-Text Matching (https://arxiv.org/pdf/1909.02701.pdf).
# The code has been modified from python2 to python3.
# -----------------------------------------------------------------

import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=False, dropout=0.5)



        # self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers,
        #                           dropout=self.rnn_dropout_p, bidirectional=bidirectional, batch_first=True)

        # result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
        #                  self.dropout, self.training, self.bidirectional, self.batch_first)
        #
        # result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
        #                  self.num_layers, self.dropout, self.training, self.bidirectional)
        #
        # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        #
        # self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
        #                    bidirectional=True, dropout=0.5)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()

        #print(dim_vid)

        vid_feats = self.vid2hid(vid_feats.contiguous().view(-1, dim_vid))
        #vid_feats = self.vid2hid(vid_feats.reshape(-1, dim_vid))
        #print(vid_feats.shape)

        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        #print(vid_feats.shape)
        #vid_feats = vid_feats.reshape(seq_len, batch_size, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden
