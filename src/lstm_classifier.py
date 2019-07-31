import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim) #TODO aca le podemos meter layers, dropout, batch_first

        # The linear layer that maps from hidden state space to tag space
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores
