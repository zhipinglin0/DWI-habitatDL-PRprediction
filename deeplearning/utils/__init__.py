import os
import sys
import json
import pickle
import math
import numpy as np

import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import openpyxl
import matplotlib.pyplot as plt


def roc_auc(trues, preds):
    fpr, tpr, thresholds = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    return fpr, tpr, auc
def matrix_myself(y_true, y_pred):
    ziped1 = zip(y_true, y_pred)
    ziped2 = zip(y_true, y_pred)
    ziped3 = zip(y_true, y_pred)
    ziped4 = zip(y_true, y_pred)
    tp = sum([(a == 1) & (b == 1) for (a, b) in ziped1])
    fn = sum([(a == 1) & (b == 0) for (a, b) in ziped2])
    fp = sum([(a == 0) & (b == 1) for (a, b) in ziped3])
    tn = sum([(a == 0) & (b == 0) for (a, b) in ziped4])

    matrix = np.array([[tp, fp], [fn, tn]])
    return (matrix)

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return (optimal_th)




@torch.no_grad()
def evaluate(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    val_trues, val_preds,val_preds_arg = [], [],[]
    val_midpre = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, labels = data
        labels = labels.view(-1, 1)
        sample_num += images1.shape[0]  # batch-size

        pred = model(images1.to(device))
        '''
        待增加功能：输出pred入excel，最好是整个验证集一起输出而不是一个batchsize输出一次
        '''

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        val_trues.extend(labels.detach().cpu().numpy())
        # val_preds.extend(torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy())
        val_preds.extend(torch.sigmoid(pred).detach().cpu().numpy())

        val_midpre.extend(pred.detach().cpu().numpy())

        loss = loss_function(pred, labels.to(device).float())
        accu_loss += loss
    threshold = ROC(val_trues, val_preds)
    val_preds_arg = [1 if t > threshold else 0 for t in val_preds]

    val_acc = accuracy_score(val_trues, val_preds_arg)


    confu_matrix = matrix_myself(val_trues, val_preds_arg)

    sensitivity = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[1, 0])
    specificity = confu_matrix[1, 1] / (confu_matrix[0, 1] + confu_matrix[1, 1])
    f1_score = metrics.f1_score(val_trues, val_preds_arg)
    Recall = metrics.recall_score(val_trues, val_preds_arg)
    Precision = metrics.precision_score(val_trues, val_preds_arg)
    positive_prediction = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[0, 1])
    negative_prediction = confu_matrix[1, 1] / (confu_matrix[1, 1] + confu_matrix[1, 0])

    fpr, tpr, AUC = roc_auc(val_trues, val_preds)

    confu_matrix = confusion_matrix(val_trues, val_preds_arg)

    tn, fp, fn, tp = confu_matrix.ravel()
 
    val_loss = accu_loss.item() / (step + 1)
    return threshold,val_loss,AUC



@torch.no_grad()
def evaluate_test(model, data_loader, device,threshold):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    val_trues, val_preds,val_preds_arg = [], [],[]
    val_midpre = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, labels = data
        labels = labels.view(-1, 1)
        sample_num += images1.shape[0]  # batch-size
        pred = model(images1.to(device))
        '''
        待增加功能：输出pred入excel，最好是整个验证集一起输出而不是一个batchsize输出一次
        '''

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        val_trues.extend(labels.detach().cpu().numpy())
        # val_preds.extend(torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy())
        val_preds.extend(torch.sigmoid(pred).detach().cpu().numpy())

        val_midpre.extend(pred.detach().cpu().numpy())

        loss = loss_function(pred, labels.to(device).float())
        accu_loss += loss

    val_preds_arg = [1 if t > threshold else 0 for t in val_preds]

    val_acc = accuracy_score(val_trues, val_preds_arg)


    confu_matrix = matrix_myself(val_trues, val_preds_arg)

    sensitivity = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[1, 0])
    specificity = confu_matrix[1, 1] / (confu_matrix[0, 1] + confu_matrix[1, 1])
    f1_score = metrics.f1_score(val_trues, val_preds_arg)
    Recall = metrics.recall_score(val_trues, val_preds_arg)
    Precision = metrics.precision_score(val_trues, val_preds_arg)
    positive_prediction = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[0, 1])
    negative_prediction = confu_matrix[1, 1] / (confu_matrix[1, 1] + confu_matrix[1, 0])

    fpr, tpr, AUC = roc_auc(val_trues, val_preds)

    confu_matrix = confusion_matrix(val_trues, val_preds_arg)

    tn, fp, fn, tp = confu_matrix.ravel()
    val_loss = accu_loss.item() / (step + 1)
    return threshold,val_loss,AUC



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


