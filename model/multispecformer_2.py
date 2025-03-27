import torch
from torch import nn
import torch.nn.functional as F
import argparse
from MulT_transformer import *
    
class MultispecFormer(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct MultispecFormer model.
        """
        super(MultispecFormer, self).__init__()
        self.orig_d_s1, self.orig_d_s2, self.orig_d_s3 = hyp_params['orig_d_s1'], hyp_params['orig_d_s2'], hyp_params['orig_d_s3']
        self.d_s1, self.d_s2, self.d_s3 = hyp_params['d_s1'], hyp_params['d_s2'], hyp_params['d_s3']
        self.num_heads = hyp_params['num_heads']
        self.layers = hyp_params['layers']
        self.attn_dropout = hyp_params['attn_dropout']
        self.attn_dropout_s2 = hyp_params['attn_dropout_s2']
        self.attn_dropout_s3 = hyp_params['attn_dropout_s3']
        self.relu_dropout = hyp_params['relu_dropout']
        self.res_dropout = hyp_params['res_dropout']
        self.out_dropout = hyp_params['out_dropout']
        self.embed_dropout = hyp_params['embed_dropout']
        self.attn_mask = hyp_params['attn_mask']
        self.use_crossmodal = hyp_params['use_crossmodal']
        self.crossmodal_first = hyp_params['crossmodal_first']
        self.s1, self.s2, self.s3 = hyp_params['s1'], hyp_params['s2'], hyp_params['s3']
        assert sum([self.s1, self.s2, self.s3])>0, 'Cannot build the model.'
        output_dim = hyp_params['output_dim']       # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        
        if sum([self.s1, self.s2, self.s3]) == 3:
            self.trans_s1_mem = self._make_encoder(self_type='s1_mem', layers=3)
            self.trans_s2_mem = self._make_encoder(self_type='s2_mem', layers=3)
            self.trans_s3_mem = self._make_encoder(self_type='s3_mem', layers=3)
            self.tokenizer_1 = SpecTokenizer(self.orig_d_s1, self.d_s1) 
            self.tokenizer_2 = SpecTokenizer(self.orig_d_s2, self.d_s2) 
            self.tokenizer_3 = SpecTokenizer(self.orig_d_s3, self.d_s3) 
            if self.use_crossmodal:
                # if self.crossmodal_first:
                self.trans_s1_with_s2 = self._make_encoder(self_type='s1_s2')
                self.trans_s1_with_s3 = self._make_encoder(self_type='s1_s3')
                self.trans_s2_with_s1 = self._make_encoder(self_type='s2_s1')
                self.trans_s2_with_s3 = self._make_encoder(self_type='s2_s3')
                self.trans_s3_with_s1 = self._make_encoder(self_type='s3_s1')
                self.trans_s3_with_s2 = self._make_encoder(self_type='s3_s2')
                combined_dim = 2 * (self.d_s1 + self.d_s2 + self.d_s3)
            else: combined_dim = self.d_s1 + self.d_s2 + self.d_s3
        elif sum([self.s1, self.s2, self.s3]) == 2:
            if self.s1 and self.s2:
                combined_dim = self.d_s1 + self.d_s2
                self.orig_d_s1, self.orig_d_s2 = self.orig_d_s1, self.orig_d_s2
                self.d_s1, self.d_s2 = self.d_s1, self.d_s2
                self_type_12, self_type_21 = 's1_s2', 's2_s1'
                self_type_1, self_type_2 = 's1_mem', 's2_mem'
            elif self.s1 and self.s3:
                combined_dim = self.d_s1 + self.d_s3              
                self.orig_d_s1, self.orig_d_s2 = self.orig_d_s1, self.orig_d_s3   
                self.d_s1, self.d_s2 = self.d_s1, self.d_s3   
                self_type_12, self_type_21 = 's1_s3', 's3_s1'
                self_type_1, self_type_2 = 's1_mem', 's3_mem'
            elif self.s2 and self.s3:
                combined_dim = self.d_s3 + self.d_s2
                self.orig_d_s1, self.orig_d_s2 = self.orig_d_s2, self.orig_d_s3             
                self.d_s1, self.d_s2 = self.d_s2, self.d_s3             
                self_type_12, self_type_21 = 's2_s3', 's3_s2'
                self_type_1, self_type_2 = 's2_mem', 's3_mem'
            if self.use_crossmodal:
                self.trans_s1_with_s2 = self._make_encoder(self_type=self_type_12)
                self.trans_s2_with_s1 = self._make_encoder(self_type=self_type_21)
            self.trans_s1_mem = self._make_encoder(self_type=self_type_1, layers=3)
            self.trans_s2_mem = self._make_encoder(self_type=self_type_2, layers=3)
            # elif self.s1 and self.s3:
            #     combined_dim = self.d_s1 + self.d_s3
            #     self.d_s1, self.d_s2 = self.d_s1, self.d_s3
            #     if self.use_crossmodal:
            #         self.trans_s1_with_s2 = self._make_encoder(self_type='s1_s3')
            #         self.trans_s2_with_s1 = self._make_encoder(self_type='s3_s1')
            #     self.trans_s1_mem = self._make_encoder(self_type='s1_mem', layers=3)
            #     self.trans_s2_mem = self._make_encoder(self_type='s3_mem', layers=3)
            # elif self.s2 and self.s3:
            #     combined_dim = self.d_s3 + self.d_s2
            #     self.d_s1, self.d_s2 = self.d_s2, self.d_s3
            #     if self.use_crossmodal:
            #         self.trans_s1_with_s2 = self._make_encoder(self_type='s2_s3')
            #         self.trans_s2_with_s1 = self._make_encoder(self_type='s3_s2')
            #     self.trans_s1_mem = self._make_encoder(self_type='s2_mem', layers=3)
            #     self.trans_s2_mem = self._make_encoder(self_type='s3_mem', layers=3)
            self.tokenizer_1 = SpecTokenizer(self.orig_d_s1, self.d_s1) 
            self.tokenizer_2 = SpecTokenizer(self.orig_d_s2, self.d_s2)
        elif sum([self.s1, self.s2, self.s3]) == 1:
            if self.s1:
                combined_dim = self.d_s1
                self.orig_d_s, self.d_s = self.orig_d_s1, self.d_s1
                self.trans_mem = self._make_encoder(self_type='s1_mem', layers=3)
            if self.s2:
                combined_dim = self.d_s2
                self.orig_d_s, self.d_s = self.orig_d_s2, self.d_s2
                self.trans_mem = self._make_encoder(self_type='s2_mem', layers=3)
            if self.s3:
                combined_dim = self.d_s3
                self.orig_d_s, self.d_s = self.orig_d_s3, self.d_s3
                self.trans_mem = self._make_encoder(self_type='s3_mem', layers=3)
            self.tokenizer = SpecTokenizer(self.orig_d_s, self.d_s)
        else: raise ValueError("No data was used.")
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def _make_encoder(self, self_type='s1', layers=-1):
        if self_type in ['s1', 's2_s1', 's3_s1']:
            embed_dim, attn_dropout = self.d_s1, self.attn_dropout
        elif self_type in ['s2', 's1_s2', 's3_s2']:
            embed_dim, attn_dropout = self.d_s2, self.attn_dropout_s2
        elif self_type in ['s3', 's1_s3', 's2_s3']:
            embed_dim, attn_dropout = self.d_s3, self.attn_dropout_s3
        if sum([self.s1, self.s2, self.s3]) == 3:
            if self.use_crossmodal:
                factor = 2 if self.crossmodal_first else 1
                if self_type == 's1_mem':
                    embed_dim, attn_dropout = factor*self.d_s1, self.attn_dropout
                elif self_type == 's2_mem':
                    embed_dim, attn_dropout = factor*self.d_s2, self.attn_dropout
                elif self_type == 's3_mem':
                    embed_dim, attn_dropout = factor*self.d_s3, self.attn_dropout
            else:
                if self_type == 's1_mem':
                    embed_dim, attn_dropout = self.d_s1, self.attn_dropout
                elif self_type == 's2_mem':
                    embed_dim, attn_dropout = self.d_s2, self.attn_dropout
                elif self_type == 's3_mem':
                    embed_dim, attn_dropout = self.d_s3, self.attn_dropout               
        elif sum([self.s1, self.s2, self.s3]) == 2:
            if not self.s3:
                if self_type == 's1_mem':
                    embed_dim, attn_dropout = self.d_s1, self.attn_dropout
                elif self_type == 's2_mem':
                    embed_dim, attn_dropout = self.d_s2, self.attn_dropout         
            elif not self.s2:
                if self_type == 's1_mem':
                    embed_dim, attn_dropout = self.d_s1, self.attn_dropout
                elif self_type == 's3_mem':
                    embed_dim, attn_dropout = self.d_s3, self.attn_dropout  
            else:
                if self_type == 's2_mem':
                    embed_dim, attn_dropout = self.d_s2, self.attn_dropout
                elif self_type == 's3_mem':
                    embed_dim, attn_dropout = self.d_s3, self.attn_dropout   
        elif sum([self.s1, self.s2, self.s3]) == 1:
            if self.s1:
                embed_dim, attn_dropout = self.d_s1, self.attn_dropout 
            elif self.s2:
                embed_dim, attn_dropout = self.d_s2, self.attn_dropout 
            elif self.s3:
                embed_dim, attn_dropout = self.d_s3, self.attn_dropout
             
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
    
    def bicrossmodal(self, x_s1, x_s2):
        proj_x_s1 = self.tokenizer_1(x_s1)
        proj_x_s2 = self.tokenizer_2(x_s2)
        if self.crossmodal_first:
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
        else:
            h_s1s = self.trans_s1_mem(proj_x_s1)
            h_s2s = self.trans_s2_mem(proj_x_s2)
            h_s1_with_s2s = self.trans_s1_with_s2(h_s1s, h_s2s, h_s2s)[-1]
            h_s2_with_s1s = self.trans_s2_with_s1(h_s2s, h_s1s, h_s1s)[-1]

            last_hs = torch.cat([h_s1_with_s2s, h_s2_with_s1s], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output

    def multicrossmodal(self, x_s1, x_s2, x_s3):
        proj_x_s1 = self.tokenizer_1(x_s1)
        proj_x_s2 = self.tokenizer_2(x_s2)
        proj_x_s3 = self.tokenizer_3(x_s3)
        if self.crossmodal_first:
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
            
        else:
            h_s1s = self.trans_s1_mem(proj_x_s1)
            h_s2s = self.trans_s2_mem(proj_x_s2)
            h_s3s = self.trans_s2_mem(proj_x_s2)
            h_s1_with_s2s = self.trans_s1_with_s2(h_s1s, h_s2s, h_s2s)[-1]
            h_s2_with_s1s = self.trans_s2_with_s1(h_s2s, h_s1s, h_s1s)[-1]
            h_s1_with_s3s = self.trans_s1_with_s3(h_s1s, h_s3s, h_s3s)[-1]
            h_s3_with_s1s = self.trans_s3_with_s1(h_s3s, h_s1s, h_s1s)[-1]
            h_s3_with_s2s = self.trans_s3_with_s2(h_s3s, h_s2s, h_s2s)[-1]
            h_s2_with_s3s = self.trans_s2_with_s3(h_s2s, h_s3s, h_s3s)[-1]

            last_hs = torch.cat([h_s1_with_s2s, h_s2_with_s1s, h_s1_with_s3s, h_s3_with_s1s, h_s2_with_s3s, h_s3_with_s2s], 1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)            


        return output
    
    def singlespec(self, x_s, tokenizer=None, trans_mem=None):
        if sum([self.s1, self.s2, self.s3]) == 3:
            tokenizer, trans_mem = tokenizer, trans_mem
        else:
            tokenizer, trans_mem = self.tokenizer, self.trans_mem
        proj_x_s = tokenizer(x_s)
        h_s = trans_mem(proj_x_s)
        if type(h_s) == tuple:
            h_s = h_s[0]
        last_hs = h_s[-1]
        if self.use_crossmodal:
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
            last_hs_proj += last_hs
            
            output = self.out_layer(last_hs_proj) 
        else: output = last_hs
        return output        

    def multispec(self, x_s1, x_s2, x_s3):
        last_hs = torch.cat([self.singlespec(x_s1, self.tokenizer_1, self.trans_s1_mem), self.singlespec(x_s2, self.tokenizer_2, self.trans_s2_mem), self.singlespec(x_s3, self.tokenizer_3, self.trans_s3_mem)], -1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output     
    
    def forward(self, inputs):
        if len(inputs) == 3:
            if self.use_crossmodal:
                output = self.multicrossmodal(inputs[0], inputs[1], inputs[2])
            else:
                output = self.multispec(inputs[0], inputs[1], inputs[2])
        elif len(inputs) == 2:
            output = self.bicrossmodal(inputs[0], inputs[1])
        else:
            output = self.singlespec(inputs[0])
        return output

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mutiple spectroscopic methods identification')
    parser.add_argument('-f', default='', type=str)

    # Fixed
    parser.add_argument('--model', type=str, default='MulT',
                        help='name of the model to use (Transformer, etc.)')

    # Tasks
    parser.add_argument('--s1', default=1, type=bool,
                        help='use the crossmodal fusion into s3 (default: False)')
    parser.add_argument('--s2', default=1, type=bool,
                        help='use the crossmodal fusion into s2 (default: False)')
    parser.add_argument('--s3', default=0, type=bool,
                        help='use the crossmodal fusion into s1 (default: False)')
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
    parser.add_argument('--use_crossmodal', type=bool, default=1,
                        help='Whether to use crossmodal transformer module.')
    parser.add_argument('--crossmodal_first', type=bool, default=0, 
                        help='use crossmodal transformer module.')
    
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
    hyp_params = {
    'no_kfval':0,
    'attn_dropout': 0.1,
    'attn_dropout_s2':0.0,
    'attn_dropout_s3':0.0,
    'relu_dropout':0.1,
    'embed_dropout':0.25,
    'res_dropout':0.1,
    'out_dropout':0.0,
    'layers':6,
    'num_heads' :8,
    'attn_mask':0,
    'orig_d_s1': 900,
    'orig_d_s2': 993,
    'orig_d_s3': 999,
    'layers': 6,
    'nlevels': 6,
    'output_dim': 957,
    'd_s1':64,
    'd_s2':64,
    'd_s3':64,
}
    hyp_params['orig_d_s1'], hyp_params['orig_d_s2'], hyp_params['orig_d_s3'] = 900, 993, 999
    hyp_params['output_dim'] = 957
    hyp_params['s1'], hyp_params['s2'], hyp_params['s3'] = args.s1, args.s2, args.s3
    hyp_params['attn_mask'] = args.attn_mask
    hyp_params['use_crossmodal'] = args.use_crossmodal
    hyp_params['crossmodal_first'] = args.crossmodal_first
    model= MultispecFormer(hyp_params)
    xlist = [torch.randn(16, 900), torch.randn(16, 993), torch.randn(16, 999)]
    if_use = [hyp_params['s1'], hyp_params['s2'], hyp_params['s3']]
    inputs = [xlist[i] for i in range(len(xlist)) if if_use[i]]

    from torch.utils.tensorboard.writer import SummaryWriter
    tb_writer = SummaryWriter(log_dir = 'net')
    tb_writer.add_graph(model, (inputs,))

    from thop import profile
    # from time import time
    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')