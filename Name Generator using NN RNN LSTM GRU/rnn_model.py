import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from dataset_utils import get_dataset, english_names, tamil_names
from train_utils import run_model, generate, save_model

class NameGeneratorRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,output_size,num_layers=1):
        super(NameGeneratorRNNModel,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_size, num_layers, batch_first=True, nonlinearity='tanh',dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, X, lengths, hidden):
        embedded = self.embedding(X)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
BATCH_SIZES = 256
dataset,train_loader, test_loader = get_dataset(BATCH_SIZES, data="tamil")
vocab_size = len(dataset.characters)+1
hidden_size = 200
num_layers= 2
embedding_dim = 15
output_size = vocab_size


if __name__ == "__main__":
    rnn_model = NameGeneratorRNNModel(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.005)    
    run_model(rnn_model, rnn_optimizer, 20, train_loader, test_loader, criterion, vocab_size)
    generated_names = generate(rnn_model, start_str='.', iterations=20, dataset=dataset, names=tamil_names)
    print(generated_names)
    save_model(rnn_model, "rnn_model")