import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,z_dim=16,hidden_dim=32,action_dim=4,num_layers=1):
        super(RNN,self).__init__()

        self.LSTM=nn.LSTM(
            input_size=z_dim+action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self,z,action,hidden=None):
        x=torch.cat([z,action],dim=-1)
        x=x.unsqueeze(1)
        out,hidden=self.LSTM(x,hidden) # out = so this is the memory collected till that time step , hidden = hidden state for that current time step
        out=out.squeeze(1)
        return out, hidden



class MDN(nn.Module):
    def __init__(self,input_dim,output_dim,k):
        super(MDN,self).__init__()
        self.k=k
        self.output_dim=output_dim

        self.pi=nn.Linear(input_dim,k)
        self.mu=nn.Linear(input_dim,k*output_dim)
        self.log_var=nn.Linear(input_dim,k*output_dim)

    def forward(self,x):
        pi=self.pi(x)
        pi=F.softmax(pi,dim=-1)
        mu=self.mu(x)
        mu=mu.view(-1,self.k,self.output_dim)
        log_var=self.log_var(x)
        log_var=log_var.view(-1,self.k,self.output_dim)
        log_var=torch.exp(0.5*log_var)

        return pi,mu,log_var


class MDRNN(nn.Module):
    def __init__(self,z_dim,action_dim,hidden_dim,k):
        super(MDRNN,self).__init__()

        self.hidden_dim=hidden_dim
        self.z_dim=z_dim
        self.k=k
        self.rnn=RNN(z_dim,hidden_dim,action_dim)
        self.mdn=MDN(hidden_dim,z_dim,k)

    def forward(self,z,action,hidden=None):
        rnn_out,hidden=self.rnn(z,action,hidden)
        pi,mu,log_var=self.mdn(rnn_out)
        return pi,mu,log_var,hidden


def mdn_loss(pi,mu,log_var,target):
    target=target.unsqueeze(1).expand_as(mu)
    var=log_var
    prob = torch.exp(-0.5 * ((target - mu) ** 2) / var) / torch.sqrt(2 * torch.pi * var)  #pdf gaussian distribution
    prob=torch.prod(prob,dim=2)
    weighted_prob=pi*prob
    nll = -torch.log(torch.sum(weighted_prob, dim=1) + 1e-8) #-ve log likelihood
    return torch.mean(nll)



'''
#dummy example
z=torch.randn(1,16)
action=torch.zeros(1,4)
action[0,2]=1    # for jst picking an action 2

rnn=RNN()
z_next=rnn(z,action)
print(z_next)

z = torch.randn(1, 16)
action = torch.zeros(1, 4)
action[0, 2] = 1

mdnrnn = MDRNN(z_dim=16, action_dim=4, hidden_dim=32, k=5)
pi, mu, log_var, hidden = mdnrnn(z, action)
print(pi.shape, mu.shape, log_var.shape)
'''