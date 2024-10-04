# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pickle
from typing import Callable, List, Optional, Tuple, Union
# import util.logging as logging
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp
from timm.layers.format import Format, nchw_to
from timm.layers.trace_utils import _assert

# logger = logging.get_logger(__name__)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
                
        return x


class PatchEmbed_HS_MAE(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        with open('./util/hibert/hilbert_5.pkl', 'rb') as f:
            self.curve = pickle.load(f)
        self.embed_dim = embed_dim
        self.num_patches = 1024
        

    def forward(self, x):
        B, C, H, W = x.shape

        # if self.img_size is not None:
        #     _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        #     _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        # x = self.proj(x)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        # elif self.output_fmt != Format.NCHW:
        #     x = nchw_to(x, self.output_fmt)
        # x = self.norm(x)
        
        '''hibert'''
        x_1D = Hibertcurve_Sample(x, self.curve)
        _, _, patchnum = x_1D.shape  # patchnum = 1024
        x_1D = x_1D.view(B, C, -1, 1).view(B,C,-1, 1, 1).transpose(2,1).flatten(0,1).cuda()
        # print(x_1D.shape)
        x = self.proj(x_1D)
        patchnum = 1024
        self.num_patches = patchnum
        x = x.view(B, patchnum, -1)
        x = self.norm(x)
        # print('=============', x.shape)
        # x = x.view(B,-1,patchnum,self.embed_dim)
        # print('======********', x.shape)
                
        return x







class PatchEmbed_HS_mae(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=1,
        in_chans=3,
        embed_dim=384,
        norm_layer: Optional[Callable] = None,
        # temporal related:
        frames=32,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        # kernel_size = [t_patch_size] + list(patch_size)
        # self.proj = nn.Conv3d(
        #     in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        # )
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        with open('./util/hibert/hilbert_5.pkl', 'rb') as f:
            self.curve = pickle.load(f)
        self.embed_dim = embed_dim

    def forward(self, x):
        # print(self.grid_size, self.t_grid_size)  14 1
        B, C, T, H, W = x.shape
        # print(x.shape) # torch.Size([2, 3, 16, 224, 224])
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        # print(x.shape) # torch.Size([2, 3, 1, 224, 224])
        
        '''hibert'''
        x = torch.einsum("bcthw->btchw", x)
        x = x.view(B*T, C, H, W)
        # print(x.shape)  # torch.Size([2, 3, 224, 224])
        x_1D = Hibertcurve_Sample(x, self.curve)
        _, _, patchnum = x_1D.shape  # patchnum = 1024
        x_1D = x_1D.view(B, C, -1, 1).view(B,C,-1, 1, 1).transpose(2,1).flatten(0,1).cuda()
        x = self.proj1(x_1D)
        patchnum = 1024
        self.num_patches = patchnum
        x = x.view(B, patchnum, -1)
        x = self.norm(x)
        # print(x2.shape)
        x = x.view(B,-1,patchnum,self.embed_dim)
        # print(x.shape) :torch.Size([B, 1, 1024, 384])
        # x = x[:, :, :196, :]
        
        # x = self.proj(x).flatten(3)
        # print('222', x.shape)
        # # print(x.shape)  # torch.Size([B, 1024, 8, 196])
        # x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        # print(x.shape)  # torch.Size([B, 8, 196, 1024])
        # assert 0
        # print('patchembedresult:', x.shape)  #torch.Size([2, 1, 196, 384])
        
        x_1D = Hibertcurve_Sample(x, self.curve)
        _, _, patchnum = x_1D.shape  # patchnum = 1024
        x_1D = x_1D.view(B, C, -1, 1).view(B,C,-1, 1, 1).transpose(2,1).flatten(0,1).cuda()
        x = self.proj1(x_1D)
        patchnum = 1024
        self.num_patches = patchnum
        x = x.view(B, patchnum, -1)
        x = self.norm(x)
        # print(x2.shape)
        x = x.view(B,-1,patchnum,self.embed_dim)
        
        
        
        
        return x


class PatchEmbed_HS(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 384,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.grid_size = self.grid_size[0]
        self.t_grid_size = 1   # 这里再看看
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        with open('./util/hibert/hilbert_5.pkl', 'rb') as f:
            self.curve = pickle.load(f)
        self.num_patches = 1024
        self.embed_dim = embed_dim
        
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = torch.einsum("bcthw->btchw", x)
            x = x.view(b*t,c,h,w)
            print('dim',b, c, t, h, w)
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x_1D = Hibertcurve_Sample(x, self.curve)  # torch.Size([1, 3, 224, 224]) ->1D: torch.Size([1, 3, 4096])
        # x_1D = x_1D.view(B, C, -1, 16 * 16).transpose(1, 2).flatten(2)
        # print(x_1D.shape)
        # 五阶 1024  六阶 4096  七阶 16384
        # x_1D = x_1D.view(B, C, -1, 1).permute(0, 2, 3, 1).flatten(2)  # 不再以16*16*3为一个patch，而是以一个三通道的像素点为一个patch
        _, _, patchnum = x_1D.shape  # patchnum = 1024
        x_1D = x_1D.view(B, C, -1, 1).view(B,C,-1, 1, 1).transpose(2,1).flatten(0,1).cuda()  # [8, 3, 4096]->[32768, 3, 1, 1]
        x = self.proj1(x_1D)
        # print(x.shape)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        # elif self.output_fmt != Format.NCHW:
        #     x = nchw_to(x, self.output_fmt)

        self.num_patches = patchnum
        x = x.view(B, self.num_patches, -1)
        x = self.norm(x)
        x = x.view(B,-1,patchnum,self.embed_dim)
        print('------------------', x.shape)
        print('+++++++++++++++++++', B,-1,patchnum,self.embed_dim)
        print('patchembedresult:', x.shape)  #torch.Size([2, 1, 196, 384])
        return x


def Hibertcurve_Sample(img, cumulative_curve):
    # with open(curve, 'rb') as f:
    #     cumulative_curve = pickle.load(f)
    n = cumulative_curve.shape[0]
    B, C, _, _ = img.shape
    # curve_len = cumulative_curve.shape[0]
    # target_num = int(curve_len/196)*196  # 196
    # start_index = np.random.randint(low=0, high=curve_len-target_num)
    # sample_curve = cumulative_curve[start_index:start_index+target_num, :]
    new_img = torch.zeros((B, C, len(cumulative_curve)), dtype=torch.float32)
    for i, coord in enumerate(cumulative_curve):
        x, y = coord
        new_img[:, :, i] = img[:, :, y, x]

    return new_img



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        self.dim = dim
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.dim = dim

    def forward(self, x):
        # print(self.dim)  # 384
        # print(x.shape)   # torch.Size([2, 513, 384])
        # print(self.norm1(x))  # torch.Size([2, 513, 384])报错  torch.Size([2, 99, 384])可以
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
