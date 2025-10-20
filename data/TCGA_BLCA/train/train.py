import sys, argparse, os, copy, itertools, glob, datetime
import random
import pickle
import time
from collections import OrderedDict

import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file

import timm
import timm.optim
import timm.scheduler

# with open('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/misc_files/uuid_TMB_labels.pkl','rb') as f:
#     uuid_TMB = pickle.load(f)
#     uuid_label_dict = dict(zip(uuid_TMB['dir_uuid'],uuid_TMB['TMB_H/L']))

with open('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/misc_files/uuid_SlideID_TMB.pkl','rb') as f:
    uuid_slideId_TMB = pickle.load(f)
    slideid_label_dict = dict(zip(uuid_slideId_TMB['slide_id'],uuid_slideId_TMB['TMB']))
    uuid_slideId_dict = dict(zip(uuid_slideId_TMB['uuid'],uuid_slideId_TMB['slide_id']))
    uuid_label_dict = dict(zip(uuid_slideId_TMB['uuid'],uuid_slideId_TMB['TMB']))

# load data
'''
features:
{'index0': ('d2e43ec6-5027-4f2c-932b-28a681da7cd9',
  'H',
  tensor([[6.0179e-01, 5.3835e-03, 2.5720e-01,  ..., 7.8925e-01, 0.0000e+00,
           5.0090e-02],
          [6.4110e-01, 4.6946e-02, 1.7235e-01,  ..., 1.0013e+00, 2.6936e-03,
           7.4646e-03],
          [3.6412e-01, 2.1077e-01, 1.2262e-01,  ..., 1.5276e+00, 0.0000e+00,
           1.3891e-02],
          ...,
          [1.0418e+00, 7.6830e-02, 4.8469e-01,  ..., 1.3794e+00, 0.0000e+00,
           6.5968e-02],
          [1.2909e+00, 8.3095e-04, 4.8256e-01,  ..., 3.1031e-01, 0.0000e+00,
           0.0000e+00],
          [4.8334e-01, 3.1399e-01, 1.1323e-01,  ..., 7.9585e-01, 4.0617e-01,
           4.0914e-03]])),}
'''

def timer(func):
    def wrapper(*args, **kw):
        t1=time.time()
        # 这是函数真正执行的地方
        result = func(*args, **kw)
        t2=time.time()

        # 计算下时长
        cost_time = t2-t1 
        print(f"{func.__name__}花费时间：{cost_time}秒")
        return result
    return wrapper

def trshuffle(tensor):
    indices = torch.randperm(tensor.size(0))
    shuffled_tensor = torch.index_select(tensor, 0, indices)
    return shuffled_tensor

def pdshuffle(tensor):
    shuffled_tensor = shuffle(pd.DataFrame(tensor))
    return shuffled_tensor

@timer
def load_split(scale,version,K):
    # version choose from:[history,config]
    
    train = np.load(f'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/{version}/data_segmentation_csv/{scale}X_tv_grouping.npy',allow_pickle=True).item()
    
    train_list = []
    val_list = []
    test_list = train['test_list']['dir_uuid'].tolist()

    if K != 0:
        X = train['tv_list']['dir_uuid']
        y = train['tv_list']['TMB_H/L']
        sKF = StratifiedKFold(n_splits=K)  #这里不需要设置random state，因为shuffle=False
        for train_iloc,val_iloc in sKF.split(X,y):
            train_list.append([X.iloc[i] for i in train_iloc])
            val_list.append([X.iloc[i] for i in val_iloc])  # 返回5个是用于val的uuid_list
    else:
        # 这里应该加载  config/data_segmentation_csv/{scale}X_grouping.npy 文件，获取train / val的分组
        pass

    return train_list,val_list,test_list

@timer
def load_feats(extractor,scale):
    feature_path = f'/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder_threshold_80/features/size512/{extractor}/{scale}X_features.pkl'
    with open(feature_path,'rb') as f:
        feats = pickle.load(f)
    return feats

'''
{'TCGA-GC-A3WC-01Z-00-DX1.D8F5CD43-7338-414C-ADE8-AC0BBC6A871C': {'features': tensor([[1.0813, 0.1505, 0.0632,  ..., 2.0810, 0.0000, 0.0501],
        [0.6613, 0.1021, 0.5238,  ..., 1.9399, 0.0000, 0.3100],
        [0.8163, 0.1676, 0.0970,  ..., 1.4184, 0.0029, 0.1104],
        ...,
        [0.9917, 0.3442, 0.4310,  ..., 1.2672, 0.0000, 0.0732],
        [1.3654, 0.0540, 0.0899,  ..., 0.8043, 0.0369, 0.3624],
        [0.5926, 0.0260, 0.0685,  ..., 1.3088, 0.0000, 0.0652]]),
'coords': array([[ 2,  8],
        [ 2,  9],
        [ 2, 10],
        [ 2, 34],
        [ 2, 35],
        [ 2, 36],
'''

@timer
def set_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    print('seed set')
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# @timer
def get_bag_feats(features,uuid,args):
    label_dict = {'L': 0, 'H': 1}
    slide_id = uuid_slideId_dict[uuid]
    try:
        feats_og = features[slide_id]['features']
        feats = np.array(pdshuffle(feats_og))
        label_og = label_dict[uuid_label_dict[uuid]]  # transformed label in form of int,[0,1]

        label = np.zeros(args.num_classes)
        
        if args.num_classes == 1:
            label[0] = label_og
        else:
            if int(label_og) <= (len(label) - 1):
                label[int(label_og)] = 1
        return label, feats
    except:
        print('invalid data, may be removed during reduce')

# @timer
def train(features, train_list, milnet, criterion, optimizer,args):
    milnet.train()
    train_list = shuffle(train_list)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    #     Tensor = torch.FloatTensor
    for i in range(len(train_list)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(features,train_list[i],args)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = Variable(Tensor(np.array([label])))
        bag_feats = Variable(Tensor(np.array([feats])))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)  # 使用多卡会出现多组bag prediciton，使计算bag loss时criterion时bag prediction / bag labelshape不匹配 -- 单卡多线程
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5 * bag_loss + 0.5 * max_loss
        l2_loss = 0
        if args.reg == True:
            for param in milnet.parameters():
                l2_loss += torch.norm(param)
        loss += args.reg_coef*l2_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    #         sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))
    return total_loss / len(train_list)


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

# @timer
def test(features,test_list, milnet, criterion,args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    #     Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(len(test_list)):
            label, feats = get_bag_feats(features,test_list[i],args)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_prediction = torch.mean(bag_prediction, dim=0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
            l2_loss = 0
            if args.reg == True:
                for param in milnet.parameters():
                    l2_loss += torch.norm(param)
            loss += args.reg_coef*l2_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal, fpr, tpr, precision, recall = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_list)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_list)

    return total_loss / len(test_list), avg_score, auc_value, thresholds_optimal,fpr, tpr, precision, recall


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        precision, recall, thr_pr = precision_recall_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal, fpr, tpr, precision, recall

best_score = 0
best_auc = 0
best_avg_auc = 0
best_loss = 0
best_loss_score = 0
best_loss_auc = 0
best_loss_avg_auc = 0
best_loss_epoch = 0

def recording(record,net,saving_path,current_score,aucs,test_loss_bag,epoch,thresholds_optimal,fpr,tpr, precision,recall):
    global best_score
    global best_auc
    global best_avg_auc
    global best_loss
    global best_loss_score
    global best_loss_auc
    global best_loss_avg_auc
    global best_loss_epoch
    # return:

    def model_recorder(record,net,thresholds_optimal,metrics,saving_path,fpr,tpr, precision,recall,epoch):
        # DONOT need model_recording when using K fold cross-validation，if needed, set arg:recording
        # arg::record only affect model saving, training curve will be recorded anyway
        # record should be ['none','best','all']
        # if record == none, weight will not be saved, else, weights will be saved with name generated by recorder every epoch after conditional judgement
        save_name = None
        if record == 'none':
            pass
        elif record == 'best':
            save_name = os.path.join(saving_path, f'{metrics}.pth')
            save_threshold = os.path.join(saving_path, f'{metrics}.pkl')
            save_auc = os.path.join(saving_path, f'{metrics}_auc.pkl')
            save_pr = os.path.join(saving_path, f'{metrics}_pr.pkl')
        elif record == 'all':
            save_name = os.path.join(saving_path, str(epoch)+f'{metrics}.pth')
            save_threshold = os.path.join(saving_path, str(epoch)+f'{metrics}.pth')
        if save_name:
            torch.save(net.state_dict(), save_name)
            with open(save_threshold, 'wb') as f:
                pickle.dump(thresholds_optimal, f)
            with open(save_auc,'wb') as f:
                pickle.dump([fpr,tpr], f)
            with open(save_pr,'wb') as f:
                pickle.dump([precision,recall], f)
    # print infos with or w/o saving models
    if current_score >= best_score:
        best_score = current_score
        print('Best_score thresholds ===>>> ' + '|'.join(
            'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        model_recorder(record,net,thresholds_optimal,'best_score',saving_path,fpr,tpr, precision,recall,epoch)

    if aucs[1] >= best_auc:
        best_auc = aucs[1]
        print('Best_auc thresholds ===>>> ' + '|'.join(
            'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        print(f'Best auc ===>>>{best_auc}')
        model_recorder(record, net,thresholds_optimal,'best_auc', saving_path, fpr,tpr, precision,recall,epoch)

#     if sum(aucs) >= best_avg_auc:
#         best_avg_auc = sum(aucs)
#         print('Best_avg_auc thresholds ===>>> ' + '|'.join(
#             'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
#         print('best avg_auc ===>>>' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))
#         model_recorder(record, net, 'best_avg_auc', saving_path, epoch)

    if best_loss == 0:
        best_loss = test_loss_bag
    if test_loss_bag < best_loss:
        best_loss = test_loss_bag
        best_loss_score = current_score
        best_loss_auc = aucs[1]
        best_loss_epoch = epoch
        model_recorder(record, net, thresholds_optimal, 'best_loss', saving_path, fpr,tpr, precision,recall,epoch)
    return best_score,best_auc,best_avg_auc,best_loss_epoch,best_loss_score,best_loss_auc,best_loss


def recording_single(record,net,saving_path,current_score,aucs,test_loss_bag,epoch,thresholds_optimal,fpr,tpr,precision, recall):
    global best_score
    global best_auc
    global best_avg_auc
    global best_loss
    global best_loss_score
    global best_loss_auc
    global best_loss_avg_auc
    global best_loss_epoch
    # return:

    def model_recorder(record,net,thresholds_optimal,metrics,saving_path,fpr,tpr,precision, recall,epoch):
        # DONOT need model_recording when using K fold cross-validation，if needed, set arg:recording
        # arg::record only affect model saving, training curve will be recorded anyway
        # record should be ['none','best','all']
        # if record == none, weight will not be saved, else, weights will be saved with name generated by recorder every epoch after conditional judgement
        save_name = None
        if record == 'none':
            pass
        elif record == 'best':
            save_name = os.path.join(saving_path, f'{metrics}.pth')
            save_threshold = os.path.join(saving_path, f'{metrics}.pkl')
            save_auc = os.path.join(saving_path, f'{metrics}_auc.pkl')
            save_pr = os.path.join(saving_path, f'{metrics}_pr.pkl')
        elif record == 'all':
            save_name = os.path.join(saving_path, str(epoch)+f'{metrics}.pth')
            save_threshold = os.path.join(saving_path, str(epoch)+f'{metrics}.pth')
        if save_name:
            torch.save(net.state_dict(), save_name)
            with open(save_threshold, 'wb') as f:
                pickle.dump(thresholds_optimal, f)
            with open(save_auc,'wb') as f:
                pickle.dump([fpr,tpr], f)
            with open(save_pr,'wb') as f:
                pickle.dump([precision,recall], f)
    # print infos with or w/o saving models
    if current_score >= best_score:
        best_score = current_score
        print('Best_score thresholds ===>>> ' + 'class-1{}'.format(thresholds_optimal))
        model_recorder(record,net,thresholds_optimal,'best_score',saving_path,fpr,tpr, precision,recall,epoch)

    if aucs[0] >= best_auc:
        best_auc = aucs[0]
        print('Best_auc thresholds ===>>> ' + 'class-1{}'.format(thresholds_optimal))
        print(f'Best auc ===>>>{best_auc}')
        model_recorder(record, net,thresholds_optimal,'best_auc', saving_path, fpr,tpr, precision,recall,epoch)
        

#     if sum(aucs) >= best_avg_auc:
#         best_avg_auc = sum(aucs)
#         print('Best_avg_auc thresholds ===>>> ' + '|'.join(
#             'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
#         print('best avg_auc ===>>>' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))
#         model_recorder(record, net, 'best_avg_auc', saving_path, epoch)

    if best_loss == 0:
        best_loss = test_loss_bag
    if test_loss_bag < best_loss:
        best_loss = test_loss_bag
        best_loss_score = current_score
        best_loss_auc = aucs
        best_loss_epoch = epoch
        model_recorder(record, net, thresholds_optimal, 'best_loss', saving_path,fpr,tpr, precision,recall, epoch)
    return best_score,best_auc,best_avg_auc,best_loss_epoch,best_loss_score,best_loss_auc,best_loss

# @timer
def train_epoch(k,features,train_list,val_list,real_test_list,milnet,criterion,optimizer,args,scheduler):

    saving_folder = f'lrwdT_{args.lr}_{args.weight_decay}_{args.Tmax}_{args.dropout_patch}_{args.dropout_node}'
    writer = SummaryWriter(f'run/{saving_folder}/fold{k}')  # path saving tensorboard logdir
    save_path = os.path.join('weights', f'{saving_folder}/fold{k}')  # path saving models
    os.makedirs(save_path, exist_ok=True)
    
    if args.record == 'none':   
        save_path = ''

    for epoch in range(1, args.num_epochs):
        train_list = shuffle(train_list)
        val_list = shuffle(val_list)
        
        train_loss_bag = train(features,train_list, milnet, criterion, optimizer,args)  # iterate all bagsload
        test_loss_bag, avg_score, aucs, thresholds_optimal,fpr, tpr, precision, recall = test(features, val_list, milnet, criterion, args)
        
        if epoch%5 ==0 or epoch <10:
            real_test_loss_bag, test_avg_score, test_aucs,_ ,_,_,_,_ = test(features, real_test_list, milnet, criterion, args)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
            'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
        if args.scheduler:
            if args.warmup:
                scheduler.step(epoch)
            else:
                scheduler.step()
        else:
            pass
        current_score = (sum(aucs) + avg_score) / 2  # 均衡考虑auc与accuracy
        
        #training curve will be recorded anyway
        if args.num_classes == 2:
            writer.add_scalar('learning rate',optimizer.param_groups[0]["lr"],epoch)
            writer.add_scalar('train_loss', train_loss_bag, epoch)
            writer.add_scalar('val_loss', test_loss_bag, epoch)
            writer.add_scalar('avg_score', avg_score, epoch)
            writer.add_scalar('class 0 aucs', aucs[0], epoch)
            writer.add_scalar('class_1 aucs', aucs[1], epoch)

            writer.add_scalar('test_loss', real_test_loss_bag, epoch)
            writer.add_scalar('test_avg_score', test_avg_score, epoch)
            writer.add_scalar('test_class_1 aucs', test_aucs[1], epoch)
            best_score, best_auc, best_avg_auc, best_loss_epoch, best_loss_score, best_loss_auc, best_loss= recording(args.record, milnet, save_path, current_score, aucs, test_loss_bag, epoch, thresholds_optimal,fpr,tpr, precision,recall)
        else:
            writer.add_scalar('train_loss', train_loss_bag, epoch)
            writer.add_scalar('val_loss', test_loss_bag, epoch)
            writer.add_scalar('avg_score', avg_score, epoch)
            writer.add_scalar('class_1 aucs', aucs[0], epoch)

            writer.add_scalar('test_loss', real_test_loss_bag, epoch)
            writer.add_scalar('test_avg_score', test_avg_score, epoch)
            writer.add_scalar('test_class_1 aucs', test_aucs[0], epoch)
            best_score, best_auc, best_avg_auc, best_loss_epoch, best_loss_score, best_loss_auc, best_loss= recording_single(args.record, milnet, save_path, current_score, aucs, test_loss_bag, epoch, thresholds_optimal,fpr,tpr, precision,recall)  # 这里保存的是current score，和test acc不同，test acc保存在tensorboard 的avg_score中


    return best_score, best_auc, best_avg_auc, best_loss_epoch, best_loss_score, best_loss_auc, best_loss


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

@timer
def reload_model(args):
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    
    if args.extractor in ['pretrained_resnet18','pretrained_resnet50']:
        weight = torch.load('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/train/init_weights/new_init_512.pth')
    elif args.extractor == 'retccl_res50_2048':
        weight = torch.load('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/train/init_weights/new_init_2048.pth')
    elif args.extractor == 'cTransPath':
        weight = torch.load('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/train/init_weights/new_init_768.pth')
    else:
        weight = torch.load('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/train/init_weights/new_init_256.pth')


    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    milnet = torch.nn.DataParallel(milnet)
    milnet = milnet.cuda()

    if args.num_classes == 2:
        milnet.load_state_dict(weight,strict=True)
    else:
        weight = torch.load('/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder/train/init_weights/init.pth')
        milnet.load_state_dict(weight,strict=False)
    print('model reloaded')
    return milnet

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    if v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    # to finetune
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--Tmax', default=50, type=int, help='Tmax used in CosineAnnealingLR,choose from [200,100,50]')
    
    # select features
    parser.add_argument('--scale',type=int,help='select magnificant scale of the extracted patches,select form [10,20]')
    parser.add_argument('--extractor',type=str,default='pretrained_resnet18',help='choose proper feature extractor')
    
    #to explore
    parser.add_argument('--reg',type=bool,default=False,help='add extra l2 regularization to loss [False]')
    parser.add_argument('--reg_coef',type=float,default=0,help='regularization coeffcient added to the punishment')
    parser.add_argument('--scheduler',type=str2bool,default=True,help='whether use CosineAnnealing scheduler')
    parser.add_argument('--warmup',type=str2bool,default=False,help='add warmup to cosineannealing scheduler')
    
    # fixed args
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=str, help='GPU ID(s) [0]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=str2bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--description', type=str,
                        help='short description for the trial, saving results in file: results_{description}.txt')
    parser.add_argument('--record', type=str, default='best',
                        help='activate tensorboard and record, save the best model, result file will be generated anyway,choose from [best,all,none][best]')
    parser.add_argument('--kfold',type=int,default=0,help='use k fold cross val')
    args = parser.parse_args()
    #     gpu_ids = args.gpu_index
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    set_seed()

    VERSION = 'history'

    # load data
    train_list,val_list,test_list = load_split(args.scale,version=VERSION,K=args.kfold) # list of 5 list
    features = load_feats(args.extractor,args.scale)

    print(f'load split: scale:{args.scale}, read in folder:{VERSION}, {args.kfold} folds in total')
    print(f'load features: scale:{args.scale}, extractor:{args.extractor}')

    # saving results during each epoch
    metrics = {
        'score' : [],
        'auc' : [],
        'best_epoch': [],
        'loss_score':[],
        'loss_auc':[],
        'loss':[]
    }

    for i in range(len(train_list)):
        #重载模型、criterion、optimizer、scheduler
        milnet = reload_model(args)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        if args.warmup:
            scheduler = timm.scheduler.CosineLRScheduler(optimizer,args.Tmax,lr_min=1e-8,warmup_t=args.Tmax//10,warmup_lr_init=1e-5)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.Tmax, 0.000005)
            
        print(f'fold:{i+1}')

        sc,auc,_,best_ep,loss_sc,loss_auc,best_ls= train_epoch(i,features,train_list[i],val_list[i],test_list,milnet,criterion,optimizer,args,scheduler)

        metrics['score'].append(sc)
        metrics['auc'].append(auc)
        metrics['best_epoch'].append(best_ep)
        metrics['loss_score'].append(loss_sc)
        metrics['loss_auc'].append(loss_auc)
        metrics['loss'].append(best_ls)
        print(f'fold{i+1} finished')
        
    avg_score = np.mean(metrics['score'])
    avg_auc = np.mean(metrics['auc'])
    avg_best_epoch = np.mean(metrics['best_epoch'])
    avg_loss_score = np.mean(metrics['loss_score'])
    avg_loss_auc = np.mean(metrics['loss_auc'])
    avg_loss = np.mean(metrics['loss'])
    with open(f'results_{args.description}.txt', 'a+') as f:
        if args.dropout_patch or args.dropout_node:
            f.write(f'{args.lr},{args.weight_decay},{args.Tmax},{args.dropout_patch},{args.dropout_node},{avg_score},{avg_auc},{avg_best_epoch},{avg_loss_score},{avg_loss_auc},{avg_loss}\n')
        else:
            f.write(f'{args.lr},{args.weight_decay},{args.Tmax},{avg_score},{avg_auc},{avg_best_epoch},{avg_loss_score},{avg_loss_auc},{avg_loss}\n')
    print(metrics['auc'])       
    print(metrics['loss_auc'])

if __name__ == '__main__':
    tick = time.time()
    main()
    print(time.time() - tick)

