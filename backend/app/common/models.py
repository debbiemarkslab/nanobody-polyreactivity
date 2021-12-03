import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=20,out_channels=32,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,stride=3))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,stride=1))
        self.linear = torch.nn.Linear(896, 1, bias=False) # 1024 is 128 channels * 8 width
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
#         print(x.shape)
        out = self.conv1(x)
#         print(out.shape)
        out = self.conv2(out)
#         print(out.shape)
        out = self.conv3(out)
#         print(out.shape)
        out = out.view(x.shape[0],out.shape[1]*out.shape[2])
        out = self.linear(out)
#         print(out.shape)
        out = self.sigmoid(out)
#         print(out.shape)
        return out

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        
        out = self.sigmoid(out)
        return out