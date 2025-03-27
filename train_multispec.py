# conda activate multispec
# python train_multispec.py
# tensorboard --logdir=log
import os, logging, torch, argparse, config
from datetime import datetime
import torch.optim as optim
import numpy as np
# from model.MulT import MULTModel, hyp_params
from model.multispecformer_1 import MultispecFormer
# from model.MulT_2c import MULTModel, hyp_params
# from model.MulT_mlp import MULTModel, hyp_params
from utils import *
from smarts_list import label_names_extended, label_names_rs18

parser = argparse.ArgumentParser(description='FGID with multiple spectroscopic methods.')
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
parser.add_argument('--tensorboard_path', type=str, default=None,
                    help='path for tensorboard log')
parser.add_argument('--no_kfval', action='store_true',
                    help='path for tensorboard log')

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
parser.add_argument('--layers', type=int, default=6,
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
parser.add_argument('--use_crossmodal', type=int, default=True,
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
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--device', default='cuda:0',
                    help='Device')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

def main():

    args = parser.parse_args()
    hyp_params = args
    SEED=hyp_params.seed
    device=torch.device(hyp_params.device if torch.cuda.is_available() else 'cpu')
    seeding(SEED)
    filename = 'multispec_1.npz'
    data = np.load(f'dataset/{filename}')

    # ir_data, ms_data, label = np.nan_to_num(data['ir']), np.nan_to_num(data['raman']), data['label_18']
    # nmr_data = np.zeros(ir_data.shape)

    ir_data, ms_data, nmr_data, label = np.nan_to_num(data['ir']), np.nan_to_num(data['ms']), np.nan_to_num(data['nmr']), data['label_957']
    # ir_data = ms_data = np.zeros(ir_data.shape)
    hyp_params.orig_d_s1, hyp_params.orig_d_s2, hyp_params.orig_d_s3 = (i.shape[1] for i in (ir_data, ms_data, nmr_data))
    hyp_params.output_dim = label.shape[1]
    # idx = [32, 29, 26, 23, 22, 21, 19, 16, 12, 9]
    # label = np.delete(label, idx, axis=1)

    # Training Settings
    train_model_path = os.path.join('log', input('Name a folder to save result: '))
    # train_model_path = os.path.join('log', f'{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}')
    os.makedirs(train_model_path, exist_ok=True)
    learning_rate = config.STRATEGY['train']['lr']  # 1e-4
    num_epochs = config.STRATEGY['train']['epochs']  # 300
    train_batch = config.STRATEGY['train']['batch_size'] #64 CS 2023: 41  me: 1024
    val_batch = config.STRATEGY['train']['batch_size'] # me: 512
    test_batch = config.STRATEGY['test']['batch_size']
    threshold = config.STRATEGY['train']['threshold']
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(30, 300, 29), gamma=0.6)

    load=False
    patience = config.STRATEGY['train']['patience']
    tensorboard = True
    tensorboard_path = hyp_params.tensorboard_path # f'log/test'
    if label.shape[1] == 37:
        label_list = label_names_extended
    elif label.shape[1] == 18:
        label_list = label_names_rs18
    elif label.shape[1] == 957:
        label_list = [str(i+1) for i in range(957)]

    '''K-fold cross validation'''
    shuffle_splits = config.STRATEGY['train']['shuffle_splits']
    kfold_splits = config.STRATEGY['train']['kfold_splits']

    predslist = []
    y_testlist = []
    roclist = []
    mf1list = []
    mpalist = []

    dset_generator = Kfold_gen(ir_data, ms_data, nmr_data, label, SEED, shuffle_splits, kfold_splits, no_kfval=hyp_params.no_kfval)
    del data, label
    for i, (trainlist, vallist, testlist) in enumerate(dset_generator):
        '''
        Training'''
        if hyp_params.no_kfval:
            print('Training once:')
        else:
            print(f'Fold {i+1}:')
        # model = Transformer(1024, label.shape[1])
        model = MultispecFormer(hyp_params)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.STRATEGY['train']['lr'], weight_decay=config.STRATEGY['train']['weight_decay'])
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(10, 300, 10), gamma=0.85)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(30, 270, 30), gamma=0.8)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.STRATEGY['train']['CosineAnnealingLR']['T_max'], 
                                                            eta_min=config.STRATEGY['train']['CosineAnnealingLR']['eta_min'])
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=get_weight(y_train)[0])
        criterion = sigmoid_focal_loss

        logging.basicConfig(filename=os.path.join(train_model_path, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'), level=logging.INFO)

        if i==0:
            logging.info(f'Dataset: dataset/{filename}.')
            logging.info(f"Model: {model}.")
            logging.info(f'Fold {i+1} for {kfold_splits}-fold cross validation:')    
            logging.info(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
        else: pass

        os.makedirs(train_model_path+f'/{i+1}_fold', exist_ok=True)
        model = Training(trainlist, train_batch, vallist, val_batch, model, num_epochs, 
                         optimizer, criterion, threshold, train_model_path+f'/{i+1}_fold', logging,
                         lr_scheduler=lr_scheduler, load=load, patience=patience, 
                         tensorboard=tensorboard, tensorboard_path=os.path.join(tensorboard_path,f'/{i+1}_fold') if tensorboard_path is not None else None, device=device)
        '''
        Test'''
        print('Testing...')
        model = model.to(device)
        thresholds = threshold_opt(trainlist, vallist, 64, model, device=device) 

        logging.basicConfig(filename=os.path.join(train_model_path+f'/{i+1}_fold', 'log'), level=logging.INFO)
        logging.info('Test:')
        logging.info(f'Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        preds, y_test, roc, macrof1, mpa = Test(testlist, test_batch, model, thresholds, device=device)
        predslist.append(preds)
        y_testlist.append(y_test)
        roclist.append(roc)
        mf1list.append(macrof1)
        mpalist.append(mpa)

        if hyp_params.no_kfval: break
        else: continue

    plot_result(np.vstack(y_testlist), np.vstack(predslist), train_model_path, label_list)
    logging.info('Average MPA: %.4f\n' %np.mean(np.array(mpalist)))
    logging.info('Average Macro-F1: %.4f\n' %np.mean(np.array(mf1list)))


if __name__ == '__main__':
    main()