from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity, ReLU, LeakyReLU
import torch_geometric.nn as geo_nn
from torch_geometric.nn.dense import Linear
import math

class MLP_main(torch.nn.Module):
    def __init__(self,
                in_channel_size: int = 0,
                hidden_channels_size: int = 0,
                hidden_channels_amount: int = 0,
                out_channel_size: int = 0,
                dropout: float = 0.,
                batch_norm: bool = True,
                batch_norm_kwargs: Optional[Dict[str, Any]] = None,
                act_f: str = 'relu',
                act_first: bool = False,
                act_kwargs: Optional[Dict[str, Any]] = None,
                layer_bias: bool = True
                ):
        r'''A Multi-Layer Perception (MLP) model.
        '''
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channel_size, hidden_channels_size, bias=layer_bias))
        for i in range(hidden_channels_amount):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size, out_channel_size, bias=layer_bias))

        batch_norm_kwargs = batch_norm_kwargs or {}
        self.norms = torch.nn.ModuleList()
        for i in range(hidden_channels_amount+1):
            if batch_norm:
                norm = BatchNorm1d(hidden_channels_size, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)
        
        act_kwargs = act_kwargs or {}
        self.dropout = dropout
        if (act_f == 'relu'):
            self.act = ReLU(**act_kwargs)
        elif (act_f == 'lrelu'):
            self.act = LeakyReLU(**act_kwargs)
        elif (act_f is None):
            self.act = Identity()
        else:
            raise ValueError(act_f)
        self.act_first = act_first

        self.channel_list = torch.nn.ModuleList()
        self.channel_list.append(self.lins[0])
        for lin, norm in zip(self.lins[1:], self.norms):
            if (self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(norm)
            if (not self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(lin)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.lins[0](x.edge_attr)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x
    
    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.lins.__len__())
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channel_list})'
    
class MLP_flexible_act_f(torch.nn.Module):
    def __init__(self,
                in_channel_size: int = 0,
                hidden_channels_size: int = 0,
                hidden_channels_amount: int = 0,
                out_channel_size: int = 0,
                dropout: float = 0.,
                batch_norm: bool = True,
                batch_norm_kwargs: Optional[Dict[str, Any]] = None,
                act_f: list = ['relu'],
                act_first: bool = False,
                act_kwargs: Optional[Dict[str, Any]] = None,
                layer_bias: bool = True
                ):
        r'''A Multi-Layer Perception (MLP) model.
        '''
        super().__init__()

        if (len(act_f) != hidden_channels_amount+1):
            raise ValueError(f'act_channels should be the same length as hidden_channels_amount + 1 (output layer).')

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channel_size, hidden_channels_size, bias=layer_bias))
        for i in range(hidden_channels_amount):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size, out_channel_size, bias=layer_bias))

        batch_norm_kwargs = batch_norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        self.norms = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        for i in range(hidden_channels_amount+1):
            if batch_norm:
                norm = BatchNorm1d(hidden_channels_size, **batch_norm_kwargs)
            else:
                norm = Identity()
            if (act_f[i] == 'relu'):
                act = ReLU(**act_kwargs)
            elif (act_f[i] == 'lrelu'):
                act = LeakyReLU(**act_kwargs)
            elif (act_f[i] is None):
                act = Identity()
            else:
                raise ValueError(act_f[i])
            self.norms.append(norm)
            self.acts.append(act)
        
        self.dropout = dropout
        self.act_first = act_first

        self.channel_list = torch.nn.ModuleList()
        self.channel_list.append(self.lins[0])
        for lin, norm, act in zip(self.lins[1:], self.norms, self.acts):
            if (self.act_first):
                self.channel_list.append(act)
            self.channel_list.append(norm)
            if (not self.act_first):
                self.channel_list.append(act)
            self.channel_list.append(lin)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.lins[0](x)
        for lin, norm, acts in zip(self.lins[1:], self.norms, self.acts):
            if self.act_first:
                x = acts(x)
            x = norm(x)
            if not self.act_first:
                x = acts(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x
    
    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.lins.__len__())
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channel_list})'
    
class MLP_main_incl_pmf_wind(torch.nn.Module):
    def __init__(self,
                in_channel_size: int = 0,
                hidden_channels_size: int = 0,
                hidden_channels_amount: int = 0,
                out_channel_size: int = 0,
                wind_channel: Optional[int] = None,
                wind_channel_extra_size: int = 8,
                dropout: float = 0.,
                batch_norm: bool = True,
                batch_norm_kwargs: Optional[Dict[str, Any]] = None,
                act_f: str = 'relu',
                act_first: bool = False,
                act_kwargs: Optional[Dict[str, Any]] = None,
                layer_bias: bool = True
                ):
        r'''A Multi-Layer Perception (MLP) model.
        '''
        super().__init__()

        if (not wind_channel):
            wind_channel = math.floor(hidden_channels_amount/2)
        elif (wind_channel >= hidden_channels_amount):
            raise ValueError()
        self.wind_channel = wind_channel

        # Set the Linear layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channel_size, hidden_channels_size, bias=layer_bias))
        for i in range(self.wind_channel):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        # The channel where the wind data gets added
        #self.lins.append(Linear(hidden_channels_size, hidden_channels_size+wind_channel_extra_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size+wind_channel_extra_size, hidden_channels_size, bias=layer_bias))
        for i in range(self.wind_channel+1, hidden_channels_amount):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size, out_channel_size, bias=layer_bias))

        batch_norm_kwargs = batch_norm_kwargs or {}
        self.norms = torch.nn.ModuleList()
        for i in range(hidden_channels_amount+1):
            if (batch_norm and i == self.wind_channel):
                norm = BatchNorm1d(hidden_channels_size+wind_channel_extra_size, **batch_norm_kwargs)
            elif (batch_norm and i != self.wind_channel):
                norm = BatchNorm1d(hidden_channels_size, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)
        
        act_kwargs = act_kwargs or {}
        self.dropout = dropout
        if (act_f == 'relu'):
            self.act = ReLU(**act_kwargs)
        elif (act_f == 'lrelu'):
            self.act = LeakyReLU(**act_kwargs)
        elif (act_f is None):
            self.act = Identity()
        else:
            raise ValueError(act_f)
        self.act_first = act_first

        self.channel_list = torch.nn.ModuleList()
        self.channel_list.append(self.lins[0])
        for lin, norm in zip(self.lins[1:], self.norms):
            if (self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(norm)
            if (not self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(lin)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        wind_data = torch.cat((x.wind_dirs_r, x.wind_dirs_pmf), -1).repeat(x.num_edges , 1)
        x = self.lins[0](x.edge_attr)
        for i, (lin, norm) in enumerate(zip(self.lins[1:], self.norms)):
            if (i == self.wind_channel):
                x = torch.cat((x, wind_data), 1)
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x
    
    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.lins.__len__())
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channel_list})'
    
class MLP_main_incl_pdf_wind(torch.nn.Module):
    def __init__(self,
                in_channel_size: int = 0,
                hidden_channels_size: int = 0,
                hidden_channels_amount: int = 0,
                out_channel_size: int = 0,
                wind_channel: Optional[int] = None,
                wind_channel_extra_size: int = 9,
                dropout: float = 0.,
                batch_norm: bool = True,
                batch_norm_kwargs: Optional[Dict[str, Any]] = None,
                act_f: str = 'relu',
                act_first: bool = False,
                act_kwargs: Optional[Dict[str, Any]] = None,
                layer_bias: bool = True
                ):
        r'''A Multi-Layer Perception (MLP) model.
        '''
        super().__init__()

        if (not wind_channel):
            wind_channel = math.floor(hidden_channels_amount/2)
        elif (wind_channel >= hidden_channels_amount):
            raise ValueError()
        self.wind_channel = wind_channel

        # Set the Linear layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channel_size, hidden_channels_size, bias=layer_bias))
        for i in range(self.wind_channel):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        # The channel where the wind data gets added
        #self.lins.append(Linear(hidden_channels_size, hidden_channels_size+wind_channel_extra_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size+wind_channel_extra_size, hidden_channels_size, bias=layer_bias))
        for i in range(self.wind_channel+1, hidden_channels_amount):
            self.lins.append(Linear(hidden_channels_size, hidden_channels_size, bias=layer_bias))
        self.lins.append(Linear(hidden_channels_size, out_channel_size, bias=layer_bias))

        batch_norm_kwargs = batch_norm_kwargs or {}
        self.norms = torch.nn.ModuleList()
        for i in range(hidden_channels_amount+1):
            if (batch_norm and i == self.wind_channel):
                norm = BatchNorm1d(hidden_channels_size+wind_channel_extra_size, **batch_norm_kwargs)
            elif (batch_norm and i != self.wind_channel):
                norm = BatchNorm1d(hidden_channels_size, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)
        
        act_kwargs = act_kwargs or {}
        self.dropout = dropout
        if (act_f == 'relu'):
            self.act = ReLU(**act_kwargs)
        elif (act_f == 'lrelu'):
            self.act = LeakyReLU(**act_kwargs)
        elif (act_f is None):
            self.act = Identity()
        else:
            raise ValueError(act_f)
        self.act_first = act_first

        self.channel_list = torch.nn.ModuleList()
        self.channel_list.append(self.lins[0])
        for lin, norm in zip(self.lins[1:], self.norms):
            if (self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(norm)
            if (not self.act_first):
                self.channel_list.append(self.act)
            self.channel_list.append(lin)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        wind_data = torch.cat((x.kappas, x.locs_r, x.weights), -1).repeat(x.num_edges , 1)
        x = self.lins[0](x.edge_attr)
        for i, (lin, norm) in enumerate(zip(self.lins[1:], self.norms)):
            if (i == self.wind_channel):
                x = torch.cat((x, wind_data), 1)
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x
    
    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.lins.__len__())
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channel_list})'