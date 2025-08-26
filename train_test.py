# -*- coding:utf-8 -*-
import time
import numpy as np
import torch.optim.lr_scheduler
from init import reports, weight_init,binary
import torch.optim as optim
from network import Net
from torch import nn
from network import ContrastiveLoss
from torch.optim import lr_scheduler
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    total_params = count_parameters(model)
    print(f'Total parameters: {total_params}')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.numel()}')


import time
import torch.optim as optim

def train_test(
        dataset,
        train_iter,
        test_iter,
        TRAIN_SIZE,
        TEST_SIZE,
        TOTAL_SIZE,
        device,
        epoches,
        windowsize):

    train_loss_list = []
    best_oa = 0.0  
    net = Net(in_cha=154, patch=windowsize, num_class=2).to(device)  
    net.apply(weight_init)  
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    loss = ContrastiveLoss()

    print('TORAL_SIZE: ', TOTAL_SIZE)
    print('TRAIN_SIZE: ', TRAIN_SIZE)
    print('TEST_SIZE: ', TEST_SIZE)
    print('---Training on {}---\n'.format(device))

    start = time.time()  

    total_eval_time = 0  

    for epoch in range(epoches):
        train_loss_sum = 0.0
        time_epoch = time.time()  

        for step, (X1, X2, X3, X4, y) in enumerate(train_iter):
            x1 = X1.to(device)
            x2 = X2.to(device)
            x3 = X3.to(device)
            x4 = X4.to(device)
            y = y.to(device)

            y_hat1, y_hat2, y_hat3, y_hat4, y_hat = net(x1, x2, x3, x4)

            l1 = loss(y_hat1, y_hat2, y.long())
            l2 = loss(y_hat3, y_hat4, y.long())
            l = 0.5 * l1 + 0.5 * l2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()

        epoch_time = time.time() - time_epoch
        print('epoch %d, train loss %.6f, time %.2f sec' % (
            epoch + 1,
            train_loss_sum / len(train_iter.dataset),
            epoch_time))

        train_loss_list.append(train_loss_sum / len(train_iter.dataset))

        if epoch % 30 == 0:
            print('\n***Start Testing***\n')
            eval_start = time.time()
            current_oa = evaluate(test_iter=test_iter, model=net, device=device)
            eval_end = time.time()
            eval_time = eval_end - eval_start
            total_eval_time += eval_time
            print(f'***Testing Time: {eval_time:.2f} sec***\n')

            if current_oa > best_oa:
                best_oa = current_oa
                torch.save(net.state_dict(), './models/' + dataset + '_best_oa.pt')
                print(f'***Successfully Saved Best OA Model with OA = {best_oa:.4f}***\n')

    End = time.time()
    total_training_time = End - start
    print('***Training End! Total Time: %.2f sec***' % total_training_time)
    print('***Total Evaluation Time: %.2f sec (Only counted every 30 epochs)***' % total_eval_time)

    print_model_summary(net)



def evaluate(test_iter, model, device):
    classification, confusion, oa, aa, kappa, f1 = reports(test_iter, model, device=device)
    classification = str(classification)
    confusion = str(confusion)
    print(classification, confusion, oa, aa, kappa, f1)

   
    return oa
