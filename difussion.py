import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self,model:nn.Module,n_steps:int=1000,beta_start:float=1e-4,beta_end:float=0.02,device:str="cpu"):
        super(DiffusionModel,self).__init__()
        self.model=model
        self.n_steps=n_steps
        self.betas=torch.linspace(beta_start,beta_end,n_steps).to(device)
        self.alphas=1-self.betas
        self.device=device
        self.alpha_hats=torch.cumprod(self.alphas,dim=0)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        batch_size,_,_,_=x.shape
        t=torch.randint(0,self.n_steps,(batch_size,)).to(x.device)
        alpha_hat_t=self.alpha_hats[t].view(batch_size,1,1,1)
        noise=torch.randn_like(x)
        x_t=torch.sqrt(alpha_hat_t)*x+torch.sqrt(1-alpha_hat_t)*noise
        predicted_noise=self.model(x_t,t)
        loss=F.mse_loss(noise,predicted_noise)
        return loss
    
    @torch.no_grad()
    def sample(self,n_samples:int,height:int,weight:int,channels:int=3)->torch.Tensor:
        x=torch.randn(n_samples,channels,height,weight,device=self.device)
        for i in reversed(range(self.n_steps)):
            t=torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise=self.model(x,t)
            alpha=self.alphas[t].view(n_samples,1,1,1)
            alpha_hat=self.alpha_hats[t].view(n_samples,1,1,1)
            beta=self.betas[t].view(n_samples,1,1,1)
            if i>0:
                noise=torch.randn_like(x)
            else:
                noise=torch.zeros_like(x)
            x=1/torch.sqrt(alpha)*(x-(1-alpha)/torch.sqrt(1-alpha_hat)*predicted_noise)+torch.sqrt(beta)*noise
        return x