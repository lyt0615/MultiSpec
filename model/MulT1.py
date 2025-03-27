import torch
from torch import nn
import torch.nn.functional as F

from model.MulT_transformer import TransformerEncoder

import argparse
parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true', default=True,
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true', default=True,
                    help='use the crossmodal fusion into a (default: False)')
# parser.add_argument('--lonly', action='store_true', default=True,
#                     help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=6,
                    help='number of layers in the network (default: 6)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network (default: 8)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()
hyp_params = args
hyp_params.orig_d_a, hyp_params.orig_d_v = 1024, 1024
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 1024, 512, 256
hyp_params.layers = args.nlevels
hyp_params.output_dim = 18


class InputNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(InputNorm, self).__init__()
        self.features = features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.a_2.view(1, self.features, 1) * (x - mean) / (std + self.eps) + self.b_2.view(1, self.features, 1)
    

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_a, self.orig_d_v = hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_a = self.d_v = 64
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        # self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_a + self.d_v

        self.partial_mode = self.aonly + self.vonly
        # if self.partial_mode == 1:
        #     combined_dim = self.d_l   # assuming d_l == d_a == d_v
        # else:
        combined_dim = (self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.layernorm_a = InputNorm(self.d_a)
        self.layernorm_v = InputNorm(self.d_v)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_a, x_v):
        '''
        Initial input shape: [batch_size, seq_len]'''
        if len(x_a.shape) == 2:
            x_a = x_a.unsqueeze(1)
            x_v = x_v.unsqueeze(1)
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]:
        """
        # x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        # proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.layernorm_a(self.proj_a(x_a))
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.layernorm_v(self.proj_v(x_v))
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        # proj_x_l = proj_x_l.permute(2, 0, 1)
        # if self.aonly:
        # V --> A
        if self.aonly:
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = h_a_with_vs
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
        # A --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = h_v_with_as
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        # import matplotlib.pyplot as plt
        # import math
        # plt.plot(last_h_a[0].cpu().detach().numpy())
        # plt.plot(last_h_v[0].cpu().detach().numpy())
        # plt.savefig('2.png')
        if self.aonly and self.vonly:
            last_hs = torch.cat([last_h_a, last_h_v], dim=1)
        elif self.aonly:
            last_hs = last_h_a
        elif self.vonly:
            last_hs = last_h_v
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Fixed
    parser.add_argument('--model', type=str, default='MulT',
                        help='name of the model to use (Transformer, etc.)')

    # Tasks
    parser.add_argument('--vonly', action='store_true', default=True,
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_true', default=True,
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_true', default=True,
                        help='use the crossmodal fusion into l (default: False)')
    parser.add_argument('--aligned', action='store_true',
                        help='consider aligned experiment or not (default: False)')
    parser.add_argument('--dataset', type=str, default='mosei_senti',
                        help='dataset to use (default: mosei_senti)')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')

    # Tuning
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='batch size (default: 24)')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='number of chunks per batch (default: 1)')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='mult',
                        help='name of the trial (default: "mult")')
    args = parser.parse_args()
    hyp_params = args
    hyp_params.orig_d_a, hyp_params.orig_d_v = 1, 1
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 1024, 512, 256
    hyp_params.layers = args.nlevels
    hyp_params.output_dim = 18

    model = MULTModel(hyp_params)

    x1, x2, x3 = torch.randn(64, 1024, 1), torch.randn(64, 1024, 1), torch.randn(64, 1024, 1)
    output = model(x2,x3)
    print([x.shape for x in output])