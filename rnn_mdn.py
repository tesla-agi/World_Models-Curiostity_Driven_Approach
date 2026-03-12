import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self,z_dim=16,hidden_dim=32,action_dim=4,num_layers=1):
        super(RNN,self).__init__()

        self.LSTM=nn.LSTM(
            input_size=z_dim+action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc=nn.Linear(hidden_dim,z_dim)


    def forward(self,z,action,hidden=None):
        x=torch.cat([z,action],dim=-1)
        x=x.unsqueeze(1)
        out,hidden=self.LSTM(x,hidden) # out = so this is the memory collected till that time step , hidden = hidden state for that current time step
        z_next=self.fc(out.squeeze(1))

        return z_next

'''
#dummy example
z=torch.randn(1,16)
action=torch.zeros(1,4)
action[0,2]=1    # for jst picking an action 2

rnn=RNN()
z_next=rnn(z,action)
print(z_next)
'''