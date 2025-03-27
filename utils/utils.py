#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :utils.py
@Description :
@InitTime    :2024/05/10 10:23:14
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn
'''


import logging
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import make_dataloader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patientce = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score+self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        print('Model saved.')
        self.val_score = epoch_score


class Engine:
    def __init__(self, train_loader=None, eval_loader=None, test_loader=None,
                 loss_fn=None, optimizer=None, scheduler=None,
                 model=None, device='cpu', device_rank=0):

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model
        self.device = device
        self.device_rank = device_rank

        self._set_loss_fn()

    def _set_loss_fn(self):
        if self.loss_fn == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_fn == 'bce':
            self.criterion = torch.nn.BCELoss()
    
    def _put_on_device(self, data):
        if type(data) == dict:
            for k, v in data.items():
                data[k] = v.to(self.device) if type(v) == torch.Tensor else v   

        else:
            data = data.to(self.device)
        return data
    
    def train_epoch(self, epoch):

        train_losses = AverageMeter()
        self.model.train()
        bar = tqdm(self.train_loader) if self.device_rank == 0 else self.train_loader
        for batch in bar:
            data, target = batch['data'], batch['target']
            data = self._put_on_device(data)
            target = self._put_on_device(target)
            output = self.model(data)

            self.optimizer.zero_grad()
            if self.loss_fn == 'bce':
                output = torch.sigmoid(output)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item(), len(target))
            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')
        
        if self.scheduler:
            self.scheduler.step()

        if self.device_rank == 0:
            logging.info(f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')
        return train_losses.avg

    def evaluate_epoch(self, epoch):

        eval_metrics = AverageMeter()
        eval_losses = AverageMeter()

        self.model.eval()
        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader
        with torch.no_grad():
            for batch in bar:
                data, target = batch['data'], batch['target']
                data = self._put_on_device(data)
                target = self._put_on_device(target)
                output = self.model(data)

                if self.loss_fn == 'bce':
                    output = torch.sigmoid(output)
                    loss = self.criterion(output, target)
                    eval_losses.update(loss.item(), len(target))

                    prediction = torch.greater_equal(output, 0.5).to( torch.float32).cpu().detach().numpy()
                    macro_f1 = metrics.f1_score(target.cpu().detach(), prediction, 
                                                average='macro', zero_division=0.0)
                    eval_metrics.update(macro_f1.item(), len(target))

                    
                    if self.device_rank == 0:
                        bar.set_description(
                            f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f} , macro_f1:{eval_metrics.avg:6f}')

                elif self.loss_fn == 'mse':
                    if self.device_rank == 0:
                        bar.set_description(
                            f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}')

                target = target.detach().cpu().numpy()
        
        if self.device_rank == 0:
            logging.info(
                f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}, macro_f1:{eval_metrics.avg:6f}')

        return eval_losses.avg, eval_metrics.avg

    def test_epoch(self):

        test_metrics = AverageMeter()
        test_losses = AverageMeter()

        self.model.eval()
        bar = tqdm(self.test_loader) if self.device_rank == 0 else self.test_loader
        outputs = []
        predicted = []
        true = []
        with torch.no_grad():
            for batch in bar:
                data, target = batch['data'], batch['target']
                data = self._put_on_device(data)
                target = self._put_on_device(target)

                output = self.model(data)
                if self.loss_fn == 'bce':
                    output = torch.sigmoid(output)
                    prediction = torch.greater_equal(output, 0.5).to(torch.float32).cpu().detach().numpy()
                    macro_f1 = metrics.f1_score(target.cpu().detach(), prediction, 
                                                average='macro', zero_division=0.0)

                    loss = self.criterion(output, target)
                    target = target.detach().cpu().numpy()

                outputs.append(output.detach().cpu().numpy())

                test_losses.update(loss.item(), len(target))
                test_metrics.update(macro_f1.item(), len(target))
                if self.device_rank == 0:
                    bar.set_description(
                        f"test loss: {test_losses.avg:.5f} macro_f1:{test_metrics.avg:.5f}")
                predicted += prediction.tolist()
                true += target.tolist()
            outputs = np.concatenate(outputs)
        return outputs, np.array(predicted), np.array(true), test_metrics.avg, test_losses.avg


def train_model(model, lmdb_path, task, word2vec=False,
                model_save_path=None, device='cpu', ddp=False, rank=-1, **kwargs):

    from torch.optim.lr_scheduler import CosineAnnealingLR
    if rank == 0:
        writer = SummaryWriter(model_save_path.replace('checkpoints', 'runs'))

    if ddp:

        train_loader, train_sampler = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=kwargs['batch_size'], num_workers=12, mode='train', verbose=True, device=device, ddp=ddp)
        
        eval_loader = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=128, mode='eval', device=device)
        
        test_loader = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=128, mode='test', device=device)
    else:
        train_loader = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=kwargs['batch_size'], num_workers=12, mode='train', verbose=True, device=device)
        
        eval_loader = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=128, mode='eval', device=device)
        
        test_loader = make_dataloader(lmdb_path, task, word2vec=word2vec, 
                                    batch_size=128, mode='test', device=device)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])

    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # mode = 'min' if not tune else 'max'
    mode = 'max'
    es = EarlyStop(patience=kwargs['patience'], mode=mode)

    engine = Engine(train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader,
                    loss_fn='bce', optimizer=optimizer, scheduler=scheduler,
                    model=model, device=device, device_rank=rank)

    # start to train
    for epoch in range(kwargs['epoch']):
        if ddp:
            train_sampler.set_epoch(epoch)
        train_loss = engine.train_epoch(epoch)
        eval_loss, acc = engine.evaluate_epoch(epoch)
        # _, _, _, test_acc, test_loss = engine.test_epoch()

        if engine.device_rank == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('eval_loss', eval_loss, epoch)
            writer.add_scalar('val_acc', acc, epoch)
            if acc:
                es(acc, model,
                f'{model_save_path}/epoch{epoch}_acc{acc*100:.0f}.pth')
            else:
                es(eval_loss, model, f'{model_save_path}/epoch{epoch}.pth')
        if es.early_stop:
            break
    print(es.val_score)
    
    if rank == 0:
        writer.close()


def test_model(model, lmdb_path, ds, task, device='cpu', verbose=True):

    test_loader = make_dataloader(lmdb_path, task, mode='test', verbose=verbose, device=device)

    loss_fn = 'bce'
    engine = Engine(test_loader=test_loader,
                    loss_fn=loss_fn, model=model, device=device)
    outputs, pred, true, _, _ = engine.test_epoch()

    if verbose:
        from sklearn import metrics
        print(metrics.classification_report(true, pred, digits=4))
        logging.info(metrics.classification_report(true, pred, digits=4))
        logging.info(f'accuracy:{metrics.accuracy_score(true, pred):.5f}')

    return outputs, pred, true


def load_state(net, state_dict):
    # check the keys and load the weight
    net_keys = net.state_dict().keys()
    state_dict_keys = state_dict.keys()
    for key in net_keys:
        if key in state_dict_keys:
            # load the weight
            net.state_dict()[key].copy_(state_dict[key])
        else:
            print('key error: ', key)
    net.load_state_dict(net.state_dict())
    return net