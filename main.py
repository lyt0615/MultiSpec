import re
import os
import time
import json
import torch
import logging
import argparse
import config
from utils.utils import seed_everything, train_model, test_model, load_state
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def get_args_parser():
    parser = argparse.ArgumentParser(description='Mutiple spectroscopic methods identification')
    parser.add_argument('-task', default='substructure', type=str)

    parser.add_argument('--model', type=str, default='MultispecFormer',
                        help='name of the model to use (Transformer, etc.)')

    parser.add_argument('--s1', default=1, type=bool,
                        help='use the crossmodal fusion into s3 (default: False)')
    parser.add_argument('--s2', default=1, type=bool,
                        help='use the crossmodal fusion into s2 (default: False)')
    parser.add_argument('--s3', default=True, type=bool,
                        help='use the crossmodal fusion into s1 (default: False)')
    parser.add_argument('--ds', type=str, default='multispec',
                        help='dataset')

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
    parser.add_argument('--crossmodal_first', type=bool, default=1,
                        help='Whether to use crossmodal transformer module.')
    
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='batch size (default: 24)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')

    # Logistics
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--device', default='cpu',
                        help='do not use cuda')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args_parser()
    seed_everything(int(args.seed))
    ts = time.strftime('%Y-%m-%d_%H:%M', time.localtime())
    ds = args.ds
    task = args.task
    model_name = args.model
    device = args.device if not args.ddp else 'cuda'
    local_rank = 0

    if args.train or args.debug:
        mode = "train"
    elif args.tune:
        mode = "tune"
    elif args.test:
        mode = "test"

    os.makedirs(f'logs/{ds}/{task}/{model_name}', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/{ds}/{task}/{model_name}/{ts}_{mode}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO)

    logging.info({k: v for k, v in args.__dict__.items() if v})

    # ================================2. data & params ======================================
    data_root = f"datasets/{ds}"
    model_save_path = f"checkpoints/{ds}/{task}/{model_name}/{ts}"

    if args.train:
        os.makedirs(f"checkpoints/{ds}/{task}/{model_name}/{ts}", exist_ok=True)

    if task.split('_')[-1] == 'substructures':
        n_classes = 957
    else:
        raise ValueError("Dataset not found")

    params = {'net': config.NET,
              'strategy': config.STRATEGY['train'] if args.train or args.debug else config.STRATEGY['tune']}
    
    params['net']['s1'], params['net']['s2'], params['net']['s3'] = args.s1, args.s2, args.s3
    params['net']['attn_mask'] = args.attn_mask
    params['net']['use_crossmodal'] = args.use_crossmodal
    params['net']['crossmodal_first'] = args.crossmodal_first

    if 'multispec' in args.ds:
        params['net']['orig_d_s1'], params['net']['orig_d_s2'], params['net']['orig_d_s3'] = 990, 993, 999
        if_use = params['net']['s1'], params['net']['s2'], params['net']['s3']
    elif 'qm9s' in args.ds or 'exp' in args.ds:
        params['net']['orig_d_s1'], params['net']['orig_d_s2'] = 1024, 1024
        if_use = params['net']['s1'], params['net']['s2']     
    if args.batch_size:
        params['strategy']['train']['batch_size'] = int(args.batch_size)
    if args.epoch:
        params['strategy']['train']['epoch'] = int(args.epoch)
    if args.lr:
        params['strategy']['train']["lr"] = float(args.lr)

    if model_name == 'MultispecFormer':
        from model.multispecformer_2 import MultispecFormer
        model = MultispecFormer(params['net']).to(device)    
    logging.info(model)
    # ================================3. start to train/tune/test ======================================
    try:
        if args.ddp:   # set up distributed device
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(rank % torch.cuda.device_count())
            dist.init_process_group(backend="nccl")
            device = torch.device("cuda", local_rank)

            print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
            torch.multiprocessing.set_start_method('spawn')
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        base_model_path = args.base_model_path
        if base_model_path:
            model = load_state(model, torch.load(base_model_path))
            print(base_model_path)

        if args.train or args.debug:
            train_model(model, lmdb_path=ds, task=task, word2vec=args.word2vec,
                        model_save_path=model_save_path, device=device, ddp=args.ddp, rank=local_rank, **params['strategy']['train'])

        elif args.tune:
            train_model(model, save_path=f"{base_model_path}/tune", ds=args.ds,
                        device=device, fold=args.fold, tune=True, **params['strategy']['tune'])

        elif args.test:
            test_model_path = args.test_model_path
            print(test_model_path)
            model = load_state(model, torch.load(test_model_path))
            test_model(model, device=device, ds=ds, task=task, lmdb_path=ds)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        os.remove(f'logs/{ds}/{model_name}/{ts}_{mode}.log')
