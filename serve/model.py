import torch.nn as nn

class LSTMClassifier(nn.Module):
    """ LSTMClassifier class for initializing the layers for the simple 
    recurrent neural network model (RNN) used for Sentiment Analysis of 
    IMDB reviews.

    Attributes:
        embedding_dim (int) dimensionality of the embedding layer
        hidden_dim (int) dimensionality of the hidden layer(s)
        vocab_size (int) size of the vocabulary used by Bag of Words

    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):

        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None


    def forward(self, x):
        """Function to perform a forward pass of the RNN model on some 
        given input.
        
        Args:
            x (array): input used for forward propagation
        
        Returns:
            array: the next layer in the neural network produced by 
                applying the element-wise sigmoid function to the input
        
        """

        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())