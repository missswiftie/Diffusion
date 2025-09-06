import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,t_channels:int,n_groups:int=8):
        super(ResidualBlock,self).__init__()
        self.norm1=nn.GroupNorm(n_groups,in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.norm2=nn.GroupNorm(n_groups,out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if in_channels==out_channels:
            self.residual=nn.Identity()
        else:
            self.residual=nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.time_emb=nn.Linear(t_channels,out_channels)
        self.dropout=nn.Dropout(0.2)

    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        h=self.conv1(F.silu(self.norm1(x)))
        t_embed=self.time_emb(F.silu(t))[:,:,None,None]
        h+=t_embed
        h=self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h+self.residual(x)
    
class AttentionBlock(nn.Module):
    def __init__(self,n_channels:int,n_heads:int=1,d_k:int=None,n_groups:int=8):
        super(AttentionBlock,self).__init__()
        self.norm=nn.GroupNorm(n_groups,n_channels)
        self.n_heads=n_heads
        self.d_k=d_k or (n_channels//n_heads)
        self.qkv=nn.Linear(n_channels,3*n_heads*self.d_k)
        self.proj=nn.Linear(n_heads*self.d_k,n_channels)
        self.scale=self.d_k**(-0.5)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        batch_size,n_channels,height,width=x.shape
        x_reshaped=x.view(batch_size,n_channels,height*width).permute(0,2,1)
        qkv=self.qkv(x_reshaped).view(batch_size,-1,self.n_heads,3*self.d_k)
        q,k,v=torch.chunk(qkv,3,dim=-1)
        # (batch_size,l,n_heads,d_k)
        # bhld,bhjd->bhlj
        attn=torch.einsum("bihd,bjhd->bijh",q,k)*self.scale
        attn=torch.softmax(attn,dim=-2)
        out=torch.einsum("bijh,bjhd->bihd",attn,v)
        out=out.view(batch_size,-1,self.n_heads*self.d_k)
        out=self.proj(out)
        out=out.permute(0,2,1).view(batch_size,n_channels,height,width)
        return out+x
    
class DownBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,t_channels:int,n_groups:int=8,use_attention:bool=False):
        super(DownBlock,self).__init__()
        self.res=ResidualBlock(in_channels,out_channels,t_channels,n_groups)
        if use_attention:
            self.attn=AttentionBlock(out_channels)
        else:
            self.attn=nn.Identity()

    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        return self.attn(self.res(x,t))
    
class UpBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,skip_channels:int,t_channels:int,n_groups:int=8,use_attention:bool=False):
        super(UpBlock,self).__init__()
        self.res=ResidualBlock(in_channels+skip_channels,out_channels,t_channels,n_groups)
        if use_attention:
            self.attn=AttentionBlock(out_channels)
        else:
            self.attn=nn.Identity()

    def forward(self,x:torch.Tensor,skip:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        x=torch.cat([x,skip],dim=1)
        return self.attn(self.res(x,t))
    
class DownSample(nn.Module):
    def __init__(self,n_channels:int):
        super(DownSample,self).__init__()
        self.ds=nn.Conv2d(n_channels,n_channels,kernel_size=3,stride=2,padding=1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.ds(x)
    
class UpSample(nn.Module):
    def __init__(self,n_channels:int):
        super(UpSample,self).__init__()
        self.us=nn.ConvTranspose2d(n_channels,n_channels,kernel_size=4,stride=2,padding=1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.us(x)
    
class MiddleBlock(nn.Module):
    def __init__(self,n_channels:int,t_channels:int,n_groups:int=8):
        super(MiddleBlock,self).__init__()
        self.res1=ResidualBlock(n_channels,n_channels,t_channels,n_groups)
        self.attn=AttentionBlock(n_channels)
        self.res2=ResidualBlock(n_channels,n_channels,t_channels,n_groups)
    
    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        h=self.res1(x,t)
        h=self.attn(h)
        h=self.res2(h,t)
        return h
    
class TimeEmbedding(nn.Module):
    def __init__(self,n_channels:int):
        super(TimeEmbedding,self).__init__()
        self.n_channels=n_channels
        self.lin1=nn.Linear(n_channels//4,n_channels)
        self.act=nn.SiLU()
        self.lin2=nn.Linear(n_channels,n_channels)
    
    def forward(self,t:torch.Tensor):
        half_dim=self.n_channels//8
        emb=math.log(10000)/(half_dim-1)
        emb=torch.exp(torch.arange(half_dim,dtype=torch.float32,device=t.device)*(-emb))
        emb=t[:,None]*emb[None,:]
        emb=torch.cat([emb.sin(),emb.cos()],dim=-1)
        out=self.lin2(self.act(self.lin1(emb)))
        return out
    
class UNet(nn.Module):
    def __init__(self,image_channels:int=3,n_channels:int=64,
                 channel_times:list[int]=[1,2,2,4],use_attn:list[bool]=[False,False,True,True],n_blocks:int=2):
        super(UNet,self).__init__()
        self.n_sample=len(channel_times)
        self.n_blocks=n_blocks
        self.time_emb=TimeEmbedding(n_channels*4)
        self.image_proj=nn.Conv2d(image_channels,n_channels,kernel_size=3,padding=1)
        self.encoder=nn.ModuleList()
        self.decoder=nn.ModuleList()
        in_channels=n_channels
        for i in range(self.n_sample):
            out_channels=in_channels*channel_times[i]
            for j in range(self.n_blocks):
                self.encoder.append(DownBlock(in_channels,out_channels,n_channels*4,use_attention=use_attn[i]))
                in_channels=out_channels
            if i!=self.n_sample-1:
                self.encoder.append(DownSample(in_channels))
            
        self.middle=MiddleBlock(in_channels,n_channels*4)

        in_channels=out_channels
        for i in reversed(range(self.n_sample)):
            for j in range(n_blocks):
                self.decoder.append(UpBlock(in_channels,out_channels,in_channels,n_channels*4,
                                            use_attention=use_attn[i]))
            out_channels=out_channels//channel_times[i]
            skip_channel=out_channels
            self.decoder.append(UpBlock(in_channels,out_channels,skip_channel,n_channels*4
                                        ,use_attention=use_attn[i]))
            in_channels=out_channels
            if i!=0:
                self.decoder.append(UpSample(in_channels))

        self.out_norm=nn.GroupNorm(8,n_channels)
        self.act=nn.SiLU()
        self.out=nn.Conv2d(n_channels,image_channels,kernel_size=3,padding=1)

    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        t_embed=self.time_emb(t)
        h=self.image_proj(x)
        skip_connections=[h]
        for layer in self.encoder:
            if isinstance(layer,DownBlock):
                h=layer(h,t_embed)
                skip_connections.append(h)
            else:
                h=layer(h)
                skip_connections.append(h)
        h=self.middle(h,t_embed)
        for layer in self.decoder:
            if isinstance(layer,UpSample):
                h=layer(h)
            else:
                skip=skip_connections.pop()
                h=layer(h,skip,t_embed)
        h=self.out(self.act(self.out_norm(h)))
        return h