import torch
class CNN(torch.nn.Module):
    def __init__(self,input_size):
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
        self.linear = torch.nn.Linear(input_size*128, 1, bias=False) # 1024 is 128 channels * 8 width
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(x.shape[0],out.shape[1]*out.shape[2])
        out = self.linear(out)
        out = self.sigmoid(out)
        return out

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = torch.device("cpu")
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        
        out = self.sigmoid(out)
        return out