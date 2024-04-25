import torch
import torch.nn as nn
class VAautoencoder(nn.Module):
    def __init__(self,input_dim,h_dim,z_dim):
        super().__init__()
        #Rede FNN
        self.fitness = nn.Linear(z_dim,1 )

        #encoder
        self.entrada = nn.Linear(input_dim,h_dim)
        self.sigma = nn.Linear(h_dim,z_dim)
        self.mu = nn.Linear(h_dim,z_dim)

        #decode
        self.z_2hidden = nn.Linear(z_dim,h_dim)
        self.output = nn.Linear(h_dim,input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self,x):
        h = self.relu(self.entrada(x))
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu ,sigma
    
    def decode(self,z):
        h = self.relu(self.z_2hidden(z))
        output = self.sigmoid(self.output(h))
        return output
    
    def forward(self,x):
        mu,sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_reparametrizado = mu + torch.exp(sigma/2)*epsilon
        #Valor da predição
        pred_value = self.fitness(z_reparametrizado)
        x_reconstruido = self.decode(z_reparametrizado)
        return x_reconstruido,mu,sigma,pred_value



def train_model(model,optimizer,loss_func,NUM_EPOCH,dataloader,loss_per_epoch,prediction_loss):
    for epoch in range(NUM_EPOCH):
        for index, (x,y) in enumerate(dataloader):
            x_reconstructed, mu, sigma,pred = model(x)

            #Erros
            recons_loss= loss_func(x_reconstructed,x)
            kl_div = -0.5*torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            fit_error = loss_func(pred,y)
            
            optimizer.zero_grad() 
            loss_val = recons_loss + kl_div  + fit_error
            
            loss_val.backward()
            optimizer.step()
            
            loss_per_epoch[epoch] = loss_val 
            prediction_loss[epoch] = fit_error
            
            
        if epoch % 10 ==0:   
            print(f"Epoch {epoch}, Erro Treino: {loss_per_epoch[epoch]}")
            
            