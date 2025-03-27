import torch
from torch import nn
import torch.nn.functional as F
import argparse
from model.MulT_transformer import *


class MultispecFormer(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MultispecFormer, self).__init__()
        self.orig_d_s1, self.orig_d_s2, self.orig_d_s3 = hyp_params.orig_d_s1, hyp_params.orig_d_s2, hyp_params.orig_d_s3
        self.d_s1, self.d_s2, self.d_s3 = hyp_params.d_s1, hyp_params.d_s2, hyp_params.d_s3
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_s2 = hyp_params.attn_dropout_s2
        self.attn_dropout_s3 = hyp_params.attn_dropout_s3
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.no_s1, self.no_s2, self.no_s3 = hyp_params.no_s1, hyp_params.no_s2, hyp_params.no_s3
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)
        assert sum([self.no_s1, self.no_s2, self.no_s3]) >=1, 'No enough modality to build such model.'
        self.layernorm_s1 = InputNorm(self.d_s1)
        self.layernorm_s2 = InputNorm(self.d_s2)
        self.layernorm_s3 = InputNorm(self.d_s3)
        # 1. Temporal convolutional layers
        if sum([self.no_s1, self.no_s2, self.no_s3]) == 3:
            self.proj_s1 = nn.Conv1d(self.orig_d_s1, self.d_s1, kernel_size=1, padding=0, bias=False)
            self.proj_s2 = nn.Conv1d(self.orig_d_s2, self.d_s2, kernel_size=1, padding=0, bias=False)
            self.proj_s3 = nn.Conv1d(self.orig_d_s3, self.d_s3, kernel_size=1, padding=0, bias=False)
            self.layernorm_s1 = InputNorm(self.d_s1)
            self.layernorm_s2 = InputNorm(self.d_s2)
            self.layernorm_s3 = InputNorm(self.d_s3)           
            self.trans_s1_with_s2 = self.get_network(self_type='s1_s2')
            self.trans_s1_with_s3 = self.get_network(self_type='s1_s3')
            self.trans_s2_with_s1 = self.get_network(self_type='s2_s1')
            self.trans_s2_with_s3 = self.get_network(self_type='s2_s3')
            self.trans_s3_with_s1 = self.get_network(self_type='s3_s1')
            self.trans_s3_with_s2 = self.get_network(self_type='s3_s2')
            self.trans_s1_mem = self.get_network(self_type='s1_mem', layers=3)
            self.trans_s2_mem = self.get_network(self_type='s2_mem', layers=3)
            self.trans_s3_mem = self.get_network(self_type='s3_mem', layers=3)
            combined_dim = 2 * (self.d_s1 + self.d_s2 + self.d_s3)
        elif sum([self.no_s1, self.no_s2, self.no_s3]) == 2:
            if self.no_s1 and self.no_s2:
                self.orig_d_s1, self.d_s1, self.d_s2, self.d_s2 = self.orig_d_s1, self.d_s1, self.orig_d_s2, self.d_s2
                combined_dim = self.d_s1 + self.d_s2
                self.trans_s1_with_s2 = self.get_network(self_type='s1_s2')
                self.trans_s2_with_s1 = self.get_network(self_type='s2_s1')
                self.trans_s1_mem = self.get_network(self_type='s1_mem', layers=3)
                self.trans_s2_mem = self.get_network(self_type='s2_mem', layers=3)
                self.proj_s1 = nn.Conv1d(self.orig_d_s1, self.d_s1, kernel_size=1, padding=0, bias=False)
                self.proj_s2 = nn.Conv1d(self.orig_d_s2, self.d_s2, kernel_size=1, padding=0, bias=False)
                self.layernorm_s1 = InputNorm(self.d_s1)
                self.layernorm_s2 = InputNorm(self.d_s2)
            elif self.no_s1 and self.no_s3:
                self.orig_d_s1, self.d_s1, self.d_s2, self.d_s2 = self.orig_d_s1, self.d_s1, self.orig_d_s3, self.d_s3
                combined_dim = self.d_s1 + self.d_s3
                self.trans_s1_with_s2 = self.get_network(self_type='s1_s3')
                self.trans_s2_with_s1 = self.get_network(self_type='s3_s1')
                self.trans_s1_mem = self.get_network(self_type='s1_mem', layers=3)
                self.trans_s2_mem = self.get_network(self_type='s3_mem', layers=3)
                self.proj_s1 = nn.Conv1d(self.orig_d_s1, self.d_s1, kernel_size=1, padding=0, bias=False)
                self.proj_s2 = nn.Conv1d(self.orig_d_s3, self.d_s3, kernel_size=1, padding=0, bias=False)
                self.layernorm_s1 = InputNorm(self.d_s1)
                self.layernorm_s2 = InputNorm(self.d_s3)
            elif self.no_s2 and self.no_s3:
                self.orig_d_s1, self.d_s1, self.d_s2, self.d_s2 = self.orig_d_s2, self.d_s2, self.orig_d_s3, self.d_s3
                combined_dim = self.d_s2 + self.d_s3
                self.trans_s1_with_s2 = self.get_network(self_type='s2_s3')
                self.trans_s2_with_s1 = self.get_network(self_type='s3_s2')
                self.trans_s1_mem = self.get_network(self_type='s2_mem', layers=3)
                self.trans_s2_mem = self.get_network(self_type='s3_mem', layers=3)
                self.proj_s1 = nn.Conv1d(self.orig_d_s2, self.d_s2, kernel_size=1, padding=0, bias=False)
                self.proj_s2 = nn.Conv1d(self.orig_d_s3, self.d_s3, kernel_size=1, padding=0, bias=False)
                self.layernorm_s1 = InputNorm(self.d_s2)
                self.layernorm_s2 = InputNorm(self.d_s3)
        elif sum([self.no_s1, self.no_s2, self.no_s3]) == 1:
            if self.no_s1:
                combined_dim = self.d_s1
                self.orig_d, self.d = self.orig_d_s1, self.d_s1
                self.trans = self.get_network(self_type='s1')
                self.trans_mem = self.get_network(self_type='s1_mem', layers=3)
                self.proj = nn.Conv1d(self.orig_d_s1, self.d_s1, kernel_size=1, padding=0, bias=False)
                self.layernorm = InputNorm(self.d_s1)
            if self.no_s2:
                combined_dim = self.d_s2
                self.orig_d, self.d = self.orig_d_s2, self.d_s2
                self.trans = self.get_network(self_type='s2')
                self.trans_mem = self.get_network(self_type='s2_mem', layers=3)
                self.proj = nn.Conv1d(self.orig_d_s2, self.d_s2, kernel_size=1, padding=0, bias=False)
                self.layernorm = InputNorm(self.d_s2)
            if self.no_s3:
                combined_dim = self.d_s3
                self.orig_d, self.d = self.orig_d_s3, self.d_s3
                self.trans = self.get_network(self_type='s3')
                self.trans = self.get_network(self_type='s3_mem', layers=3)
                self.proj = nn.Conv1d(self.orig_d_s3, self.d_s3, kernel_size=1, padding=0, bias=False)
                self.layernorm = InputNorm(self.d_s3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='s1', layers=-1):
        if self_type in ['s1', 's2_s1', 's3_s1']:
            embed_dim, attn_dropout = self.d_s1, self.attn_dropout
        elif self_type in ['s2', 's1_s2', 's3_s2']:
            embed_dim, attn_dropout = self.d_s2, self.attn_dropout_s2
        elif self_type in ['s3', 's1_s3', 's2_s3']:
            embed_dim, attn_dropout = self.d_s3, self.attn_dropout_s3

        if self.no_s1 and self.no_s2 and self.no_s3:
            if self_type == 's1_mem':
                embed_dim, attn_dropout = 2*self.d_s1, self.attn_dropout
            elif self_type == 's2_mem':
                embed_dim, attn_dropout = 2*self.d_s2, self.attn_dropout
            elif self_type == 's3_mem':
                embed_dim, attn_dropout = 2*self.d_s3, self.attn_dropout
        elif self.no_s1 and self.no_s2 and not self.no_s3:
            if self_type == 's1_mem':
                embed_dim, attn_dropout = self.d_s1, self.attn_dropout
            elif self_type == 's2_mem':
                embed_dim, attn_dropout = self.d_s2, self.attn_dropout         
        elif self.no_s1 and self.no_s3 and not self.no_s2:
            if self_type == 's1_mem':
                embed_dim, attn_dropout = self.d_s1, self.attn_dropout
            elif self_type == 's3_mem':
                embed_dim, attn_dropout = self.d_s3, self.attn_dropout  
        elif self.no_s2 and self.no_s3 and not self.no_s1:
            if self_type == 's2_mem':
                embed_dim, attn_dropout = self.d_s2, self.attn_dropout
            elif self_type == 's3_mem':
                embed_dim, attn_dropout = self.d_s3, self.attn_dropout 

        elif self.no_s1 and not self.no_s3 and not self.no_s2:
            embed_dim, attn_dropout = self.d_s1, self.attn_dropout 
        elif self.no_s2 and not self.no_s3 and not self.no_s1:
            embed_dim, attn_dropout = self.d_s2, self.attn_dropout 
        elif self.no_s3 and not self.no_s2 and not self.no_s1:
            embed_dim, attn_dropout = self.d_s3, self.attn_dropout 

        else:
            raise ValueError("Unknown network type.")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def calc_singlespec(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        proj = x if self.orig_d == self.d else self.proj(x)
        proj = proj.permute(2, 0, 1)
        h = self.trans_mem(self.trans(proj, proj, proj))
        if type(h) == tuple:
            h = h[0]
        last_h = h[-1]        
        # A residual block
        last_h_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h)), p=self.out_dropout, training=self.training))
        last_h_proj += last_h
        output = self.out_layer(last_h_proj)
        return output
            
    def calc_bicrossmodal(self, x_s1, x_s2):
        x_s1 = x_s1.unsqueeze(1)
        x_s2 = x_s2.unsqueeze(1)
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_s1 = x_s1.transpose(1, 2)
        x_s2 = x_s2.transpose(1, 2)
       
        # Project the textual/visual/audio features
        # proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_s1 = x_s1 if self.orig_d_s1 == self.d_s1 else self.proj_s1(x_s1)   #self.layernorm_s1(self.proj_s1(x_s1))
        proj_x_s2 = x_s2 if self.orig_d_s2 == self.d_s2 else self.proj_s2(x_s2)   #self.layernorm_s2(self.proj_s2(x_s2))
        proj_x_s1 = proj_x_s1.permute(2, 0, 1)
        proj_x_s2 = proj_x_s2.permute(2, 0, 1)
        # s2 --> s1
        h_s1_with_s2s = self.trans_s1_with_s2(proj_x_s1, proj_x_s2, proj_x_s2)
        h_s1s = h_s1_with_s2s
        h_s1s = self.trans_s1_mem(h_s1s)
        if type(h_s1s) == tuple:
            h_s1s = h_s1s[0]
        last_h_s1 = last_hs = h_s1s[-1]

        # s1 --> s2
        h_s2_with_s1s = self.trans_s2_with_s1(proj_x_s2, proj_x_s1, proj_x_s1)
        h_s2s = h_s2_with_s1s
        h_s2s = self.trans_s2_mem(h_s2s)
        if type(h_s2s) == tuple:
            h_s2s = h_s2s[0]
        last_h_s2 = last_hs = h_s2s[-1]
        last_hs = torch.cat([last_h_s1, last_h_s2], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output

    def calc_multicrossmodal(self, x_s1, x_s2, x_s3):
        x_s1 = x_s1.unsqueeze(1)
        x_s2 = x_s2.unsqueeze(1)
        x_s3 = x_s3.unsqueeze(1)
        x_s1 = x_s1.transpose(1, 2)
        x_s2 = x_s2.transpose(1, 2)
        x_s3 = x_s3.transpose(1, 2)
        # x_s1 = F.dropout(x_s1.transpose(1, 2), p=self.embed_dropout, training=self.training)
       
        # Project the textual/visual/audio features
        proj_x_s1 = x_s1 if self.orig_d_s1 == self.d_s1 else self.proj_s1(x_s1)  #self.layernorm_s1(self.proj_s1(x_s1))
        proj_x_s2 = x_s2 if self.orig_d_s2 == self.d_s2 else self.proj_s2(x_s2)  #self.layernorm_s2(self.proj_s2(x_s2))
        proj_x_s3 = x_s3 if self.orig_d_s3 == self.d_s3 else self.proj_s3(x_s3)  #self.layernorm_s3(self.proj_s3(x_s3))
        proj_x_s2 = proj_x_s2.permute(2, 0, 1)
        proj_x_s3 = proj_x_s3.permute(2, 0, 1)
        proj_x_s1 = proj_x_s1.permute(2, 0, 1)

        # (s3,s2) --> s1
        h_s1_with_s2s = self.trans_s1_with_s2(proj_x_s1, proj_x_s2, proj_x_s2)    # Dimension (L, N, d_s1)
        h_s1_with_s3s = self.trans_s1_with_s3(proj_x_s1, proj_x_s3, proj_x_s3)    # Dimension (L, N, d_s1)
        h_s1s = torch.cat([h_s1_with_s2s, h_s1_with_s3s], dim=2)
        h_s1s = self.trans_s1_mem(h_s1s)
        if type(h_s1s) == tuple:
            h_s1s = h_s1s[0]
        last_h_s1 = last_hs = h_s1s[-1]   # Take the last output for prediction

        # (s1,s3) --> s2
        h_s2_with_s1s = self.trans_s2_with_s1(proj_x_s2, proj_x_s1, proj_x_s1)
        h_s2_with_s3s = self.trans_s2_with_s3(proj_x_s2, proj_x_s3, proj_x_s3)
        h_s2s = torch.cat([h_s2_with_s1s, h_s2_with_s3s], dim=2)
        h_s2s = self.trans_s2_mem(h_s2s)
        if type(h_s2s) == tuple:
            h_s2s = h_s2s[0]
        last_h_s2 = last_hs = h_s2s[-1]

        # (s1,s2) --> s3
        h_s3_with_s1s = self.trans_s3_with_s1(proj_x_s3, proj_x_s1, proj_x_s1)
        h_s3_with_s2s = self.trans_s3_with_s2(proj_x_s3, proj_x_s2, proj_x_s2)
        h_s3s = torch.cat([h_s3_with_s1s, h_s3_with_s2s], dim=2)
        h_s3s = self.trans_s3_mem(h_s3s)
        if type(h_s3s) == tuple:
            h_s3s = h_s3s[0]
        last_h_s3 = last_hs = h_s3s[-1]
        
        last_hs = torch.cat([last_h_s1, last_h_s2, last_h_s3], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output
    

    def forward(self, x_s1, x_s2, x_s3):

        if self.no_s1 and self.no_s2 and not self.no_s3:
            output = self.calc_bicrossmodal(x_s1, x_s2)
        if self.no_s1 and self.no_s3 and not self.no_s2:
            output = self.calc_bicrossmodal(x_s1, x_s3)
        if self.no_s2 and self.no_s3 and not self.no_s1:
            output = self.calc_bicrossmodal(x_s2, x_s3)
        if self.no_s1 and self.no_s2 and self.no_s3:
            output = self.calc_multicrossmodal(x_s1, x_s2, x_s3)
        if self.no_s1 and not self.no_s2 and not self.no_s3:
            output = self.calc_singlespec(x_s1)
        if self.no_s2 and not self.no_s1 and not self.no_s3:
            output = self.calc_singlespec(x_s2)
        if self.no_s3 and not self.no_s2 and not self.no_s1:
            output = self.calc_singlespec(x_s3)
        return output

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mutiple spectroscopic methods identification')
    parser.add_argument('-f', default='', type=str)

    # Fixed
    parser.add_argument('--model', type=str, default='MulT',
                        help='name of the model to use (Transformer, etc.)')

    # Tasks
    parser.add_argument('--no_s1', action='store_false',
                        help='use the crossmodal fusion into s3 (default: False)')
    parser.add_argument('--no_s2', action='store_false',
                        help='use the crossmodal fusion into s2 (default: False)')
    parser.add_argument('--no_s3', action='store_false',
                        help='use the crossmodal fusion into s1 (default: False)')
    parser.add_argument('--aligned', action='store_true',
                        help='consider aligned experiment or not (default: False)')
    parser.add_argument('--dataset', type=str, default='mosei_senti',
                        help='dataset to use (default: mosei_senti)')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_s2', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_s3', type=float, default=0.0,
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
    parser.add_argument('--d_s1', type=int, default=64,
                        help='Modality 1\'s embedding length. Has to be divided by num_heads.')
    parser.add_argument('--d_s2', type=int, default=64,
                        help='Modality 2\'s embedding length. Has to be divided by num_heads.')
    parser.add_argument('--d_s3', type=int, default=64,
                        help='Modality 3\'s embedding length. Has to be divided by num_heads.')
    
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
    hyp_params.orig_d_s1, hyp_params.orig_d_s2, hyp_params.orig_d_s3 = 900, 993, 999
    hyp_params.layers = args.nlevels
    hyp_params.output_dim = 957
    model= MultispecFormer(hyp_params)
    s1, s2, s3 = torch.randn(16, 900), torch.randn(16, 993), torch.randn(16, 999)
    print(hyp_params.no_s1, hyp_params.no_s2, hyp_params.no_s3)

    from time import time
    t1 = time()
    print(model(s1, s2, s3).shape)
    t2=time()
    print((t2-t1)/16*1000)

    from torch.utils.tensorboard.writer import SummaryWriter
    tb_writer = SummaryWriter(log_dir = '.')
    tb_writer.add_graph(model, (s1, s2, s3))

    from thop import profile
    from time import time
    flops, params = profile(model, inputs=(s1, s2, s3))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')