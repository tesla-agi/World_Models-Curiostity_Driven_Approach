import torch
import gymnasium as gym
from torch import nn,optim
import numpy as np
from vae import VAE
from rnn_mdn import RNN
from rnn_mdn import MDRNN
import time
from rnn_mdn import mdn_loss

num_eps=100
max_steps=300
input_dim=8
z_dim=16
action_dim=4
hidden_dim=32
k_mix=5
lr=1e-3

vae=VAE(input_dim=input_dim,z_dim=z_dim,hidden_dim=hidden_dim)
#rnn=RNN(z_dim=z_dim,hidden_dim=hidden_dim,action_dim=action_dim)
mdrnn=MDRNN(z_dim=z_dim,hidden_dim=hidden_dim,action_dim=action_dim,k=k_mix)

vae.eval() #freeze
for param in vae.parameters():
    vae.requires_grad=False
mdrnn.train() #train
#criterion=nn.MSELoss()
optimizer=optim.Adam(mdrnn.parameters(),lr=lr)


env=gym.make('LunarLander-v3',render_mode='human')
state=None
for episode in range(num_eps):
    state,_=env.reset()
    state=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    hidden=None
    episode_loss=0

    for t in range(max_steps):
        env.render()
        time.sleep(0.02) #slow visuals
        mu, log_var = vae.encode(state)
        z_t = vae.reparameterize(mu, log_var)
        z_t=z_t.detach() # No VAE gradients

        action_idx=np.random.randint(0,action_dim)
        action=torch.zeros(1,action_dim)
        action[0,action_idx]=1 #one hot encoding for action (which is random)

        next_state,reward,terminated,truncated,_=env.step(action_idx)
        next_state_tensor=torch.tensor(next_state,dtype=torch.float32).unsqueeze(0)

        mu_next, log_var_next = vae.encode(next_state_tensor)
        z_next = vae.reparameterize(mu_next, log_var_next)        # Ground truth
        #z_next=z_next.detach() #here also same remove the past gradients so that we can use it in RNN
        # z_next_pred=rnn(z_t,action,hidden)
        pi,mu_pred,log_var_pred,hidden=mdrnn(z_t,action,hidden)
        hidden = tuple(h.detach() for h in hidden)

        loss=mdn_loss(pi,mu_pred,log_var_pred,z_next)
        episode_loss+=loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #hidden=None
        state=next_state_tensor

        if terminated or truncated:
            break

    print(f"Episode {episode + 1}/{num_eps}  Loss: {episode_loss / (t + 1):.4f}")

env.close()