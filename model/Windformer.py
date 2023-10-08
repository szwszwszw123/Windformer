import torch
from torch import nn
from einops import rearrange
from model.Conv_Lstm import Conv_Lstm_Module,GRU_Module,RNN_Module




class GRU_Block(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size=3) -> None:
        super().__init__()
        self.front = GRU_Module(in_channels,hidden_channels,kernel_size,layers_num=1)
        
    def forward(self,x):
        front_output,_ = self.front(x)

        return front_output

class BiGRU_Block(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size=3) -> None:
        super().__init__()
        self.front = GRU_Module(in_channels,hidden_channels,kernel_size,layers_num=1)
        self.back = GRU_Module(in_channels,hidden_channels,kernel_size,layers_num=1)
        self.output_conv = nn.Conv2d(hidden_channels*2,hidden_channels,1)

    def forward(self,x):
        B = x.shape[0]
        front_output,_ = self.front(x)
        back_output,_ = self.back(x.flip(dims=[1]))
        
        back_output = back_output.flip(dims=[1])

        output = torch.cat([front_output,back_output],dim=2)
        output = self.output_conv(rearrange(output,"B T C H W -> (B T) C H W"))

        return rearrange(output,"(B T) C H W -> B T C H W",B=B)


class MS_CAM(nn.Module):
    def __init__(self,num_fea) -> None:
        super().__init__()
        self.SE = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_fea,num_fea,1),
            nn.BatchNorm2d(num_fea),
            nn.ReLU(),
            nn.Conv2d(num_fea,num_fea,1),
            nn.BatchNorm2d(num_fea)
        )
    
        self.CAM = nn.Sequential(
            nn.Conv2d(num_fea,num_fea,3,padding="same"),
            nn.BatchNorm2d(num_fea),
            nn.ReLU(),
            nn.Conv2d(num_fea,num_fea,3,padding="same"),
            nn.BatchNorm2d(num_fea)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # x (B, Id, F, Ih, Iw):
        B = x.shape[0]
        x_dim = len(x.shape)

        x = self.sigmoid(self.SE(x) + self.CAM(x))

        return x




class PatchEmbed(nn.Module):
    def __init__(self,image_size,patch_size,input_steps,num_fea,dim) -> None:
        """
        input : (B, Id, F, Ih, Iw)
        output : (B, H, W, C)
        """
        super().__init__()
        # self.has_cam = has_cam
        # if(has_cam):
        #     self.ms_cam = MS_CAM(num_fea)
        self.image_size,self.patch_size = image_size,patch_size
        self.input_steps,self.num_fea = input_steps,num_fea
        self.dim = dim

        self.proj = nn.Conv2d(input_steps * num_fea,dim,patch_size,patch_size)

    def forward(self,x):
        # B -- batch_size, C -- dim
        # x(B,input_steps,num_fea,*image_size)
        input_resolution = [self.image_size[0]//self.patch_size[0],self.image_size[1]//self.patch_size[1]]
        B = x.shape[0]

        x = rearrange(x,"B Id F Ih Iw -> B (Id F) Ih Iw")
        # x(B,C,*window_size)
        x = self.proj(x)

        # x (batch_size,window_size[0] * window_size[1],dim)
        x = from_img_to_seq(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self,input_resolution,dim) -> None:
        """
        input : (B, H, W, C)
        output : (B, H/2, W/2, C * 2)
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.proj = nn.Linear(4*dim,2*dim)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self,x):
        # H,W 指的是patch的分辨率
        # L = H * W
        # x (B, H, W, C)
        B,H,W,C = x.shape

        assert H,W == self.input_resolution

        x0 = x[:,0::2,0::2,:]
        x1 = x[:,0::2,1::2,:]
        x2 = x[:,1::2,0::2,:]
        x3 = x[:,1::2,1::2,:]

        # x (B,H/2,W/2,C*4)
        x = torch.cat([x0,x1,x2,x3],-1)
        x = self.norm(x)

        # x (B, H*W/4, C*4)
        x = x.flatten(1,2)
        x = self.proj(x)

        x = x.reshape(B,H//2,W//2,C * 2)
        return x


class WindowAttention(nn.Module):
    def __init__(self,window_size,dim,num_heads,attn_drop,proj_drop) -> None:
        """
        input x : (B_, N, C)
        input mask : mask (nW,N,N)
        output : (B_, N, C)
        """
        super().__init__()
        self.window_size,self.dim,self.num_heads = window_size,dim,num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_table = nn.Parameter(torch.zeros(((2*window_size[0]-1)*(2*window_size[1]-1),num_heads)))

        coords_h = torch.arange(0,window_size[0])
        coords_w = torch.arange(0,window_size[1])

        # coords (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h,coords_w]))
        # coords (2, Wh*Ww)
        coords = coords.flatten(1)

        # relative_coords (Wh*Ww, Wh*Ww, 2)
        relative_coords = (coords[:,:,None] - coords[:,None,:]).permute(1,2,0)
        relative_coords[:,:,0] += (window_size[0]-1) 
        relative_coords[:,:,1] += (window_size[1]-1) 
        relative_coords[:,:,0] *= (2 * window_size[1] - 1)
        # relative_position_index  (Wh*Ww, Wh*Ww)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index",relative_position_index)

        self.qkv = nn.Linear(dim,dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_table,std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask):
        # mask (nW,Wh*Ww,Wh*Ww)

        # B_:batch_size * num_windows
        # N : window 中的patch数  N = Wh * Ww
        B_,N,C = x.shape

        # nH : num_heads
        # hD : head_dim
        # q,k,v (B_, nH, N, hd)
        q,k,v = rearrange(self.qkv(x),"Bs N (nH hD qkv) -> qkv Bs nH N hD",qkv=3,nH=self.num_heads)
        
        # attn (B_, nH, N, N)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        # relative_position_bias (nH, N, N)
        relative_position_bias = rearrange(self.relative_position_table[self.relative_position_index.view(-1)],
        "(Wh0 Ww0 Wh1 Ww1) nH -> nH (Wh0 Ww0) (Wh1 Ww1)", Wh0=self.window_size[0],Wh1=self.window_size[0],Ww0=self.window_size[1],Ww1=self.window_size[1])

        attn = attn + relative_position_bias.unsqueeze(0)

        if(mask != None):    
            nW = mask.shape[0]
            attn = rearrange(attn,"(B nW) nH N0 N1 -> B nW nH N0 N1",nW=nW)
            attn = attn + rearrange(mask,"nW N0 N1 -> 1 nW 1 N0 N1")
            attn = rearrange(attn,"B nW nH N0 N1 -> (B nW) nH N0 N1")
        
        attn = self.softmax(attn)
        # x (B_,N,C)
        x = rearrange(attn @ v,"Bs nH N hd -> Bs N (nH hd)")
        return x


def from_img_to_seq(x):
    return rearrange(x, "B C H W -> B H W C")

def from_seq_to_img(x):
    return rearrange(x, "B H W C -> B C H W")

def window_partition(x,window_size):
    B,H,W,C = x.shape
    x = rearrange(x,"B (nWh Wh) (nWw Ww) C -> (B nWh nWw) (Wh Ww) C",Wh=window_size[0],Ww=window_size[1])
    return x

def window_reverse(x,window_size,H,W):
    B_,N,C = x.shape
    x = rearrange(x,"(B nWh nWw) (Wh Ww) C -> B (nWh Wh) (nWw Ww) C",Wh=window_size[0],Ww=window_size[1],nWh=H//window_size[0],nWw=W//window_size[1])
    return x


class SwinBlock(nn.Module):

    def __init__(self,dim,input_resolution,num_heads,window_size,shift_size,mlp_ratio,attn_drop,proj_drop) -> None:
        """
        input : (B, H, W, C)
        output : (B, H, W, C)
        """
        super().__init__()
        self.dim,self.input_resolution,self.num_heads = dim,input_resolution,num_heads
        self.window_size,self.shift_size = window_size,shift_size
        self.mlp_ratio,self.attn_drop,self.proj_drop = mlp_ratio,attn_drop,proj_drop

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(window_size,dim,num_heads,attn_drop,proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,dim),
            nn.ReLU()
        )

        

        if(self.shift_size[0]*self.shift_size[1] != 0):
            H,W = self.input_resolution

            mask = torch.zeros((1,H,W,1))

            h_slices = [
                slice(0,-window_size[0]),
                slice(-window_size[0],-shift_size[0]),
                slice(-shift_size[0],None)
            ]
            w_slices = [
                slice(0,-window_size[1]),
                slice(-window_size[1],-shift_size[1]),
                slice(-shift_size[1],None)
            ]
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    mask[:,h,w] = cnt
                    cnt = cnt + 1
            # mask_window (nW, Wh*Ww, C)
            mask_window = window_partition(mask,window_size)
            # mask_window (nW, Wh*Ww)
            mask_window = mask_window.reshape(mask_window.shape[0],mask_window.shape[1])
            # attn_window (nW,Wh*Ww,Wh*Ww)
            attn_mask = mask_window[:,:,None] - mask_window[:,None,:]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask",attn_mask)
        
    def forward(self,x):
        B,H,W,C = x.shape
        assert H,W == self.input_resolution

        short_cut = x

        x = self.norm1(x.reshape(B,H*W,C))
        x = x.reshape(B,H,W,C)

        if(self.shift_size[0]*self.shift_size[1] != 0):
            shifted_x = torch.roll(x,[-self.shift_size[0],-self.shift_size[1]],[1,2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x,self.window_size)
        attn_windows = self.attn(x_windows,self.attn_mask)
        # x (B,H,W,C)
        shifted_x = window_reverse(attn_windows,self.window_size,H,W)

        if(self.shift_size[0]*self.shift_size[1] != 0):
            x = torch.roll(shifted_x,self.shift_size,[1,2])
        else:
            x = shifted_x

        x = x + short_cut
        
        short_cut = x
        x = self.norm2(x.reshape(B,H*W,C))
        x = self.mlp(x)
        x = x.reshape(B,H,W,C) + short_cut

        return x




class BasicLayer(nn.Module):
    def __init__(self,dim,input_resolution,patch_size,num_heads,window_size,is_swin,has_cam,mlp_ratio=4,attn_drop=0,proj_drop=0) -> None:
        """
        input : (B, H, W, C)
        output : (B, H, W, C)
        """
        super().__init__()
        self.is_swin = is_swin
        self.has_cam = has_cam
        if(is_swin):
            self.swin = nn.Sequential(
                *[SwinBlock(dim,input_resolution,num_heads,window_size,
                [window_size[0]//2,window_size[1]//2] if(i % 2) else [0,0],mlp_ratio,attn_drop,proj_drop)
                for i in range(2)]
            )
        else:
            pass
        self.patch_merge = PatchMerge(input_resolution,dim)
        if(has_cam):
            self.ms_cam = MS_CAM(dim * 2)
            # self.norm = nn.LayerNorm(dim * 2)


    def forward(self,x):
        if(self.is_swin):
            x = self.swin(x)
        else:
            pass
        # x = self.swin(x)
        x = self.patch_merge(x)
        if(self.has_cam):
            x = from_seq_to_img(x)
            x = x * self.ms_cam(x)
            x = from_img_to_seq(x)
            # x = self.norm(x)
        return x

class Windformer(nn.Module):
    def __init__(self,image_size,patch_size,input_steps,num_fea,
    dim,window_size,depth,is_swin,has_cam,output_size,rnn=0,mlp_ratio=4,attn_drop=0,proj_drop=0) -> None:
        """
        input x : (B,input_steps,features,image_size)
        output : (B, image_size)
        """
        super().__init__()
        self.patch_size,self.image_size = patch_size,image_size
        if(rnn == 0):
            self.lstm = Conv_Lstm_Block(num_fea,num_fea,3)
        elif(rnn == 1):
            self.lstm = BiConv_Lstm_Block(num_fea,num_fea,3)
        elif(rnn == 2):
            self.lstm = GRU_Block(num_fea,num_fea,3)
        elif(rnn == 3):
            self.lstm = BiGRU_Block(num_fea,num_fea,3)
        elif(rnn == 4):
            self.lstm = RNN_Block(num_fea,num_fea,3)
        elif(rnn == 5):
            self.lstm = BiRNN_Block(num_fea,num_fea,3)
        elif(rnn == 6):
            self.lstm = None
        self.patch_embed = PatchEmbed(image_size,patch_size,input_steps,num_fea,dim)
        H,W = image_size[0]//patch_size[0],image_size[1]//patch_size[1]
        self.input_resolution = [H,W]
        self.num_fea,self.depth = num_fea,depth
        self.has_cam = has_cam
        self.layers = nn.Sequential(
            *([
                BasicLayer(dim * (2 ** i),[H // (2 ** i),W // (2 ** i)],patch_size,max(1,dim * (2 ** i) // 8),window_size,is_swin,has_cam,mlp_ratio,attn_drop,proj_drop)
                for i in range(depth)
            ])
        )

        self.output_head = nn.Linear(H * W * dim // (2 ** depth),output_size)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1)


    def forward(self,x):
        x = x[:,:,-self.num_fea:]
        # x (B, H, W, C)
        B = x.shape[0]
        if(self.lstm != None):
            x = self.lstm(x)
        x = self.patch_embed(x)
        feature_maps = []
        feature_maps.append(x)
        for i in range(self.depth):
            x = self.layers[i](x)

        x = x.reshape(B,-1)
        x = self.output_head(x)

        return x

    
