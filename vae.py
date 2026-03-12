import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,input_dim=8,z_dim=16,hidden_dim=64):
        super(VAE,self).__init__()

        #Encoder
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc_mu=nn.Linear(hidden_dim,z_dim)
        self.fc_log_var=nn.Linear(hidden_dim,z_dim)

        #Decoder
        self.fc2=nn.Linear(z_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,input_dim)

        #Activation Function
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def encode(self,x):
        h=self.relu(self.fc1(x))
        return self.fc_mu(h),self.fc_log_var(h)

    def reparameterize(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)
        return mu+std*eps                 # So this is value is for  z

    def decode(self,z):
        h=self.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self,x):
        mu,log_var=self.encode(x) # Encoded
        z=self.reparameterize(mu,log_var) # Reparameterize
        x_recon=self.decode(z)     # Reconstructed/ Decoded
        return x_recon,mu,log_var

'''
#dummy train
vae = VAE()
# Create a dummy LunarLander state (batch size 1)
dummy_state = torch.randn(1, 8)
# Forward pass
x_recon, mu, log_var = vae(dummy_state)

# Print outputs
print("Original state:", dummy_state)
print("Reconstructed state:", x_recon)
print("Latent mean:", mu)
print("Latent log-variance:", log_var)
'''
