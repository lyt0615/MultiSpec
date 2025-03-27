
# python train_singlespec.py

import os, logging, time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, sampler
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from utils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from smarts_list import label_names_extended, label_names_rs18
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from alive_progress import alive_bar, config_handler
from model.VaTransformer import VaTransformer
from model.cnn import CNN_1
from model.ResPeak import resunit
# from model.CNN_multimodal import CNN_1
from model.ResNet import resnet
from model.ResNet_MLP import resnet_mlp
# from model.cnn_1 import CNN_1
config_handler.set_global(length=80, bar='halloween')

SEED=42
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
seeding(SEED)

def Kfold_gen(data, label, SEED, shuffle_splits=1, kfold_splits=5, test_val_split=0.25):
    print(f'Performing {kfold_splits}-fold splittng...')
    # kfold cross validation
    str_shuffle = MultilabelStratifiedShuffleSplit(n_splits=shuffle_splits, test_size=test_val_split, random_state=SEED)
    str_kfold = MultilabelStratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=SEED)
    trainlist = []
    vallist = []
    testlist = []
    for temp_index, test_index in str_kfold.split(data, label):
    # for temp_index, test_index in str_shuffle.split(data, label):
        # X_temp = data[temp_index]
        # X_test = data[test_index]
        try:
            X_temp = data[temp_index]
            X_test = data[test_index]
        except:
            X_temp = [data[m] for m in temp_index]
            X_test = [data[n] for n in test_index]
        y_temp = label[temp_index]
        y_test = label[test_index]
        
        # smiles_temp = smiles[temp_index]
        # smiles_test = smiles[test_index]
        testlist = [X_test, y_test]
        for train_index, val_index in str_shuffle.split(X_temp, y_temp):
            # X_train = X_temp[train_index]
            try:
                X_train = X_temp[train_index]
                X_val = X_temp[val_index]
            except:
                X_train = [X_temp[p] for p in train_index]
                X_val = [X_temp[q] for q in val_index]
            # X_val = X_temp[val_index]
            
            y_train = y_temp[train_index]
            y_val = y_temp[val_index]
            # smiles_train = smiles_temp[train_index]
            # smiles_val = smiles_temp[val_index]
            trainlist = [X_train, y_train]
            vallist = [X_val, y_val]
            yield trainlist, vallist, testlist

def get_loader(X, y, batch_size, sampling_weight=None, shuffle=False):
    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    if sampling_weight is not None:
        weighted_sampler = sampler.WeightedRandomSampler(sampling_weight, len(sampling_weight))
    else:
        weighted_sampler = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=weighted_sampler)
    return loader

def get_weight(y, run=True):
    if run:
        fg_num = np.sum(y, axis=0)
        fg_weight = len(y)/fg_num
        sample_weight = fg_weight/np.max(fg_weight)

        sample_weight_1 = np.array([np.max(sample_weight[i.astype(bool)]) for i in y])
        return torch.tensor(sample_weight, dtype=torch.float32).to(device), torch.tensor(sample_weight_1, dtype=torch.float32).to(device)
    else: return torch.tensor(np.full(len(y), 1)).to(device), torch.tensor(np.full(len(y), 1)).to(device)

def Training(X_train, y_train, train_batch, X_val, y_val, 
             val_batch, model, num_epochs, optimizer, criterion, val_threshold, 
             train_model_path, logging, lr_decay=False, lr_scheduler=None, load=True, patience=10, tensorboard=True, tensorboard_path=None):
    
    val_loss_list = []
    train_loss_list = []
    train_loss_list_1 = []
    val_acc_list = []
    val_f1_list = []
    loss_to_class_train = []
    loss_to_class_val = []
    val_loss_min = 1
    lr_decay_count = 0

    # loss_weight_train, sampling_weight_train = get_weight(y_train)
    train_loader = get_loader(X_train, y_train, train_batch, sampling_weight=None)

    # loss_weight_val, sampling_weight_val = get_weight(y_val)
    val_loader = get_loader(X_val, y_val, val_batch, sampling_weight=None)

    if load: 
        os.makedirs(train_model_path+'/finetune', exist_ok=True)
        print('Loaded an existing model.')
        
        model.load_state_dict(torch.load(train_model_path+'/best_network.pth', map_location=device),
                                                            strict=True)
    if patience: 
        if load: 
            est = EarlyStopping(save_path=train_model_path+'/finetune', patience=10)
        else:
            est = EarlyStopping(save_path=train_model_path, patience=10)
    val_acc_max = 0.5

    if tensorboard:
        if tensorboard_path is not None: 
            tb_path = train_model_path
        else:
            tb_path = train_model_path
        os.makedirs(train_model_path, exist_ok=True)
        tb_writer = SummaryWriter(log_dir = tb_path)

    if lr_scheduler is not None:
        logging.info(f'Learning rate scheduler: {lr_scheduler}.\n{lr_scheduler.state_dict()}')
    else: 
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info(f'Learning rate: {lr}.') 
        print(f'Training started. lr={lr}.')



    logging.info(f'Saved in path: {train_model_path}.')
    logging.info(f"Optimizer: {optimizer}.")
    logging.info(f'Loss function: {criterion}.')
    logging.info(f"Learning Rate Scheduler: {lr_scheduler}.")
    logging.info(f"Number of Epochs: {num_epochs}.")
    logging.info(f"Training Batch Size: {train_batch}.")
    logging.info(f"Validation Batch Size: {val_batch}\n")

    time_started = time.time()
    for epoch in range(1, num_epochs+1):
        train_loss = 0
        train_loss_1 = 0
        model.train()
        L1_train = 0
        # train_bar = tqdm(enumerate(train_loader), total=len(train_loader))  # Create a tqdm progress bar
        with alive_bar(len(train_loader), title='Training') as train_bar:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x.to(device))
                loss = criterion(outputs, batch_y.to(device), reduction='mean')
                loss_to_class_train.append(torch.mean(criterion(outputs, batch_y.to(device), reduction='none').float().cpu(), dim=0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar(0.5)
        train_loss = train_loss / len(train_loader)
        if lr_scheduler is not None: 
            lr_scheduler.step()
        print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss}.')

        model.eval()
        val_loss = 0
        val_acc = 0
        val_f1 = 0
        num_correct = 0
        L1_val = 0
        with torch.no_grad():
            # val_bar = tqdm(enumerate(val_loader), total=len(val_loader))  # Create a tqdm progress bar
            with alive_bar(len(val_loader), title='Validation') as val_bar:
                for batch_x, batch_y in val_loader:
                    val_outputs = model(batch_x.to(device))
                    loss = criterion(val_outputs, batch_y.to(device), reduction='mean')
                    loss_to_class_val.append(torch.mean(criterion(val_outputs, batch_y.to(device), reduction='none').float().cpu(), dim=0))
                    optimizer.zero_grad()
                    predicted_labels = (val_outputs >= val_threshold).float().cpu()  # Convert to binary labels (0 or 1)
                    num_correct += np.count_nonzero(predicted_labels == batch_y) # element-wise match
                    val_f1 += metrics.f1_score(predicted_labels, batch_y, average='macro', zero_division=0)
                    val_loss += loss.item()
                    val_bar(0.5)

        val_f1_sum = val_f1/(len(y_val)//val_batch+1)
        val_acc = num_correct/(y_val.shape[0]*y_val.shape[1]) 
        val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch }/{num_epochs}], Validation Accuracy: {val_acc}.')
        # train_loss_list.append(train_loss)
        # val_loss_list.append(val_loss)
        # val_acc_list.append(val_acc)
        # val_f1_list.append(val_f1_sum)

        # if lr_decay:
            # if lr_decay_count >= 10 or val_loss <= 1e-5:
            #     current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            #     logging.info(f'Final lr={current_lr}')
            #     print('Training Finished.')
            #     break     
            # elif lr_decay_count >= 5:
            #     if optimizer.state_dict()['param_groups'][0]['lr'] >= 1.5e-4:
            #         for params in optimizer.param_groups:
            #             params['lr'] *= 0.8
            #         current_lr = params['lr']
            #         print(f'learning rate decayed to {current_lr}')

            # if val_loss < val_loss_min:
            #     val_loss_min = val_loss
            #     lr_decay_count = 0
            # else:
            #     lr_decay_count += 1
            #     print(f'Early stopping count: {lr_decay_count} / 10.')

            # Chem. Sci. 2023:
            # from net.CNN2 import custom_learning_rate_schedular
            # for params in optimizer.param_groups:
            #             params['lr'] = custom_learning_rate_schedular(epoch)
                    
        

        if tensorboard:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
            tb_writer.add_scalar("F1/val", val_f1_sum, epoch)
            tb_writer.add_graph(model.to(device), (batch_x.to(device)))

        if patience:
            est(val_loss, model)
            if est.early_stop:
                print("Early stopped.")
                time_ended=time.time()
                break
        if val_acc_max < val_acc:
            val_acc_max = val_acc
            if load: torch.save(model.state_dict(), train_model_path+'/finetune'+'/best_network.pth')
            else: torch.save(model.state_dict(), os.path.join(train_model_path, 'best_network.pth'))
            print('Validation accuracy updated. Model saved.')
        time_ended = time.time()

    if load:
        logging.info(f'Finetuning finished in {int((time_ended-time_started)//60)} mins.')
    else:
        logging.info(f'Training finished in {int((time_ended-time_started)//60)} mins.\n')

    return model

def Test(X_test, y_test, test_batch, model, thresholds=None, sigmoid=False):

    if thresholds is not None:
        print('Testing...')
    else:
        print('Optimizing thresholds...')
    test_loader = get_loader(X_test, y_test, test_batch)
    # Evaluation on test set
    preds = []
    logits = []
    y_true = []
    num_correct = 0
    with torch.no_grad():
        test_bar = tqdm(enumerate(test_loader),total=len(test_loader),ncols=150)
        for _, (batch_x, batch_y) in test_bar:
            test_outputs = model(batch_x.to(device))
            test_outputs = test_outputs.squeeze(1)
            if not sigmoid:
                test_outputs = torch.sigmoid(test_outputs)
            logits.append(test_outputs.cpu().numpy())
            # test_loss = criterion(test_outputs, batch_y.to(device))
            if thresholds is not None: preds.append(test_outputs.cpu().numpy() >= thresholds)  # Convert to binary labels (0 or 1)
            else: preds.append(test_outputs.cpu())
            y_true.append(batch_y.cpu().numpy())
            num_correct += (test_outputs.cpu() == batch_y).sum().item()
            # accuracy = (predicted_labels == batch_y).float().mean()
        preds = np.vstack(preds)
        logits = np.vstack(logits)
        y_true = np.vstack(y_true)
        if thresholds is not None:
            return preds, logits, y_true
        else: return [metrics.precision_recall_curve(y_test[:, i], preds[:, i]) for i in range(y_test.shape[1])]

def threshold_opt(X, y, opt_batch, model):

    pr_curve_train_val = Test(X, y, opt_batch, model)
    opt_threshold = []
    for element in pr_curve_train_val:
        precision = element[0]
        recall = element[1]
        threshold = element[2]
        f1_score = precision*recall/(precision+recall)
        idx = np.argmax(f1_score)
        opt_threshold.append(threshold[idx])
    print('Finished.')
    return np.array(opt_threshold)

def plot_result(y_true, y_pred, train_model_path, label_list):
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    support = support / 2
    # Sort the support array and get the sorted indices
    sorted_indices = support.argsort()

    # Rearrange the arrays based on sorted indices
    sorted_support = support[sorted_indices]
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    sorted_f1 = f1[sorted_indices]
    sorted_labels = [label_list[i] for i in sorted_indices]

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(9,4))

    # Plot 'a' and 'b' as line charts on the primary Y-axis (ax1)
    ax1.plot(sorted_labels, sorted_precision, label='Precision', color='blue', marker='o')
    ax1.plot(sorted_labels, sorted_recall, label='Recall', color='green', marker='s')
    ax1.plot(sorted_labels, sorted_f1, label='F1-score', color='red', marker='x')

    # Set labels for the primary Y-axis
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Values', color='black')

    # Create a secondary Y-axis for the histogram 'c'
    ax2 = ax1.twinx()
    ax2.bar(sorted_labels, sorted_support, color='red', alpha=0.6, width=0.4)

    # Set labels for the secondary Y-axis
    ax2.set_ylabel('Label Frequency', color='black')
    ax1.set_xticklabels(sorted_labels, rotation=90, ha='right')
    ax1.grid()

    # Add a legend
    # ax1.legend(loc = 1, bbox_to_anchor=(0.65, 0.40))
    # ax2.legend(['Frequency'], loc = 1, bbox_to_anchor=(0.497, 0.20))
    ax1.legend(bbox_to_anchor=(1.07, 1.05), loc='lower right', borderaxespad=0)
    ax2.legend(['Frequency'], bbox_to_anchor=(0.822, 1.17), loc='lower right', borderaxespad=0)
    # Show the plot
    plt.title('Performance to class distribution (Raman experimental)')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(train_model_path, 'result.png'))
    
    np.savez(os.path.join(train_model_path, 'result.npz'), f1=metrics.f1_score(y_true,y_pred,average=None),
             accuracy=np.array([metrics.accuracy_score(y_true[:,i], y_pred[:, i]) for i in range(y_true.shape[1])]))
    
def main():
    '''
    KFold cross validation'''
    data_file = 'dataset/multispec.npz' # dataset/_cut.npz  raman_exp.npz
    # data = np.load(data_file)
    # X_train = data['X_train']
    # y_train = data['y_train']
    # X_val = data['X_val']
    # y_val = data['y_val']
    # X_test = data['X_test']
    # y_test = data['y_test']
    # data = np.vstack((X_train, X_val, X_test))
    # label = np.vstack((y_train, y_val, y_test))
    # data = np.nan_to_num(data)
    dset = np.load(data_file, allow_pickle=True)
    data = dset['ir']
    # ir = dset['ir'][:, np.newaxis,:]
    # data = np.concatenate((raman, ir), axis=1)
    label = dset['label_957']

    # data = dset['raman']
    # data = np.nan_to_num(data)
    # data, label = np.vstack([data[i] for i in range(0, 120000, 12)]), np.vstack([label[k] for k in range(0, 120000, 12)])

    # CNN settings
    learning_rate = 1e-3  #2.5e-4   # 1e-4# Chem. Sci. 2023: 2.5e-4   me: 2.5e-4  の
    num_epochs = 300
    train_batch = 64 #64 CS 2023: 41  me: 1024
    val_batch = 64 # me: 512
    test_batch = 32
    threshold = 0.5
    # test_model_path = os.path.join('model',input('Name a model: '))
    train_model_path = os.path.join('log', input('Name a folder to save result: '))
    os.makedirs(train_model_path, exist_ok=True)

    load=False
    patience = False
    lr_scheduler = None
    lr_decay = False
    tensorboard = True
    tensorboard_path = f'log/test'  # None
    if label.shape[1] == 37:
        label_list = label_names_extended
    elif label.shape[1] == 18:
        label_list = label_names_rs18
    elif label.shape[1] == 957:
        label_list = [str(i+1) for i in range(957)]
    shuffle_splits = 1
    kfold_splits = 5
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(30, 300, 29), gamma=0.6)
    weight_decay = 0
    
    # trainlist, vallist, testlist = 
    predlist = []
    ytrulist = []
    macrof1list = []
    mpalist = []
    roclist = []

    dset_generator = Kfold_gen(data, label, SEED, shuffle_splits, kfold_splits)
    del data, label

    # for i in range(kfold_splits):
    for i, (trainlist, vallist, testlist) in enumerate(dset_generator):    
        '''
        Training'''
        print(f'Fold {i+1}:')
        X_train, y_train = trainlist
        X_val, y_val =  vallist
        X_test, y_test = testlist
        # model = VaTransformer(vocab_size=label.shape[1])
        # model = resnet(n_classes=label.shape[1], layers=8)
        # model = CNN_1(class_num=label.shape[1])
        model = resunit(1,957,20,6)
        # model = model = resunit(1,label.shape[1],20,6)
        # model = resnet_mlp(n_classes=label.shape[1])
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(30, 300, 30), gamma=0.8)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=get_weight(y_train)[0])
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        criterion = sigmoid_focal_loss
        logging.basicConfig(filename=os.path.join(train_model_path, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'), level=logging.INFO)
        if i==0:
            logging.info(f'Dataset: {data_file}.')
            logging.info(f"Model: {model}.")
            logging.info(f'Fold {i+1} for {kfold_splits}-fold cross validation:')    
            logging.info(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
        else: pass
        os.makedirs(train_model_path+f'/{i+1}_fold', exist_ok=True)
        print(lr_scheduler)
        model = Training(X_train, y_train, train_batch, X_val, y_val, 
                        val_batch, model, num_epochs, optimizer, criterion, threshold, train_model_path+f'/{i+1}_fold', logging,
                        lr_decay=lr_decay, lr_scheduler=lr_scheduler, load=load,
                        patience=patience, tensorboard=tensorboard, tensorboard_path=os.path.join(tensorboard_path,f'/{i+1}_fold') if tensorboard_path is not None else None)
        '''
        Test'''
        print('Testing...')
        model = model.to(device)
        thresholds = threshold_opt(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]), 64, model)        
        logging.basicConfig(filename=os.path.join(train_model_path+f'/{i+1}_fold', 'log'), level=logging.INFO)
        logging.info('Test:')
        logging.info(f'Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logging.info(f'Test data: {len(X_test)}.')
        preds, _, _ = Test(X_test, y_test, test_batch, model, thresholds)
        predlist.append(preds)
        ytrulist.append(y_test)
        # np.savez(os.path.join(train_model_path+f'/{i+1}_fold', 'output.npz'), logits=logits, preds=preds, y_true=y_true)

        # logging.info(f'Optimized Thresholds:\n{thresholds}')
        logging.info(metrics.classification_report(y_test, preds, digits=4, zero_division=0))
        # logging.info('ROC-AUC: %.4f' %metrics.roc_auc_score(y_test, preds, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None))
        logging.info('Molecular perfect accuracy: %.4f\n' %metrics.accuracy_score(y_test, preds))
        logging.info('Element-wise accuracy: %.4f\n' %(np.count_nonzero(preds == y_test)//preds.shape[0]/preds.shape[1]))
        # roclist.append(metrics.roc_auc_score(y_test, preds, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None))
        macrof1list.append(metrics.f1_score(y_test, preds, average='macro'))
        mpalist.append(metrics.accuracy_score(y_test, preds))
    plot_result(np.vstack(ytrulist), np.vstack(predlist), train_model_path, label_list)
    logging.info('Average MPA: %.4f\n' %np.mean(mpalist))
    logging.info('Average Macro-F1: %.4f\n' %np.mean(macrof1list))

if __name__ == '__main__':
    main()