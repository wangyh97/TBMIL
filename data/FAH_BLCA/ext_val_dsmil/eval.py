import sys, argparse, os, copy, itertools, glob, datetime
import logging
import random
from collections import OrderedDict
import time
import pickle

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve,f1_score,confusion_matrix
from sklearn.datasets import load_svmlight_file


def timer(func):
    def wrapper(*args, **kws):
        tick = time.time()
        func(*args, **kws)
        print(f'{func.__name__} comsumes {time.time() - tick} s')

    return wrapper


def set_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
    print('seed set')

def load_features(scale,extractor):
    with open(f'/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/features/size512/{extractor}/{scale}X_features.pkl','rb') as f:
        features = pickle.load(f)
    print(f'load features from : {extractor}/{scale}X_features.npy') 
    
    #####load only from testing sets
    
    return features

def get_label_dict():
    seq_info = pd.read_csv('/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/sequencing_information/seq_patho_comment.csv')
    seq_id = seq_info['Sample-Pair']
    TMB_numeric = seq_info['TMB']
    TMB = [i>=10 for i in TMB_numeric]
    label_dict = dict(zip(seq_id,TMB))

    return label_dict

def get_bag_feats(features,slide_id, label_dict, args):
    """
    features: features extracted from all slides using specific extractor, orginized as:
        slide_features[slide_id] = {'seq_id':seq_id,'features': feature_list}
    seq_id: seq_id of specific slide, eg. 1087
    label_dict: pair of seq_id & transformed TMB level, eg. 1087:False(0)
    """
    feats_og = features[slide_id]['features']
    seq_id = features[slide_id]['seq_id']
    label_og = label_dict[eval(seq_id)]
    feats = shuffle(feats_og)
    
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = label_og
    else:
        if int(label_og) <= (len(label) - 1):
            label[int(label_og)] = 1
    return label, feats


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(features, milnet, args):
    """
    demo of values of intermediate variables:

    original label:tensor([[0., 1.]], device='cuda:0'),shape of feats:torch.Size([1, 53, 512])
    shape of feats after view:torch.Size([53, 512])
    shape of ins_pred:torch.Size([53, 2]),original bag_pred:tensor([[-2.9899,  2.7934],
            [-3.7077,  3.5751],
            [-3.6781,  3.7495],
            [-3.8786,  3.5615]], device='cuda:0'),shape of original bag_pred:torch.Size([4, 2])
    max pred:tensor([ 0.9809, -2.4702], device='cuda:0'),bag pred after mean:tensor([-3.5636,  3.4199], device='cuda:0')
     Testing bag [0/73] bag loss: 0.9777
    test laels:[0. 1.],test prediction : [array([0.02755601, 0.9683193 ], dtype=float32)]
    first 5 class_pred_bag:[1. 0. 0. 0. 0.],
    first 5 test_pred:[[0. 1.]
     [0. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]],
     first 5 labels:[[0. 1.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]]
     first 5 single char labels:[1, 0, 0, 0, 0],first 5 single char pred:[1, 0, 0, 1, 0]
     """
    milnet.eval()

    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    #     Tensor = torch.FloatTensor

    label_dict = get_label_dict()
    with torch.no_grad():
        for slide_id in features.keys():
            label, feats = get_bag_feats(features,slide_id, label_dict, args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(feats)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_prediction = torch.mean(bag_prediction, dim=0)

            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal, fpr, tpr, precision, recall = multi_label_roc(test_labels, test_predictions,
                                                                                    args.num_classes, pos_label=1)
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
    # get confusion matrix
    one_hot_labels = [np.argmax(i) for i in test_labels]
    one_hot_preds = [np.argmax(i) for i in test_predictions]
    c = confusion_matrix(one_hot_labels, one_hot_preds)
    # tn, fp, fn, tp = c.ravel()
    f1 = f1_score(one_hot_labels, one_hot_preds)
    bag_score = 0
    for i in range(0, len(features.keys())):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(features.keys())
    return avg_score, auc_value, c, fpr, tpr, precision, recall, f1 # fpr,tpr for ROC curve plotting,precision,recall for PR-curve plotting


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


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def load_model(args, weight_path):
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    weight = torch.load(weight_path)

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                   dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    milnet = torch.nn.DataParallel(milnet)
    milnet = milnet.cuda()
    milnet.load_state_dict(weight, strict=True)
    return milnet


def weight_parser(weight_path,fold):
    path= Path(weight_path)
    metric = path.stem
    if fold == -1:
        lrwdT = path.parents[0].name
    else:
        lrwdT = path.parents[1].name
    split = lrwdT.split('_')
    if len(split) == 4:
        _, lr, weight, _ = split
        dropout_patch, dropout_node = 0,0
    elif len(split) == 6:
        _, lr, weight, _, dropout_patch, dropout_node = split
    return lr, weight, metric,  dropout_patch, dropout_node


def metrics_visulization(fpr, tpr, roc_auc, precision, recall, confusion_matrix, saving_path, figsize=(10, 20)):
    # roc curve: fpr,tpr
    # metrics
    # F1 score
    # TODO: add metrics
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    axes[0].set_title('ROC curve')
    axes[0].plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % roc_auc)
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc="lower right")

    axes[1].set_title('P-R curve')
    axes[1].plot(recall, precision, color='darkorange')
    axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1].set_xlabel('recall')
    axes[1].set_ylabel('precision')

    axes[2].set_title('confusion matrix')
    axes[2].imshow(confusion_matrix,interpolation='nearest')
    axes[2].set_xlabel('predicted label')
    axes[2].set_ylabel('true label')
    axes[2].set_xticks(np.arange(2), [0, 1])
    axes[2].set_yticks(np.arange(2), [0, 1])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

    plt.savefig(saving_path)
    plt.close()


@timer
def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')

    # select features
    parser.add_argument('--scale', type=int,help='select magnificant scale of the extracted patches,select form [10,20]')
    parser.add_argument('--extractor', type=str, default='none',help='select pretrained embedder')

    # model structure -- no use
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index',default=0, type=str, help='GPU ID(s) [0]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,help='Average the score of max-pooling and bag aggregating')

    # select the model
    parser.add_argument('--description', type=str,help='short description for the trial, saving results in file: results_{description}.txt')
    parser.add_argument('--fold', default=0, type=str, help='specific fold selected, run all models saved')
    parser.add_argument('--version', default=0, choices=['old','new'],default='old', help='specific fold selected, run all models saved')
    

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    set_seed()

    # TODO: working dir
    log_folder = os.path.join(f'/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/ext_val_dsmil',args.description)
    os.makedirs(log_folder,exist_ok=True)
    log_path = os.path.join(log_folder,'metrics.log')

    # initiate logger: show info in stream & save in file
    log = logging.getLogger('recorder')
    log.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    handler1 = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler1.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    handler1.setFormatter(formatter1)
    log.addHandler(handler)
    log.addHandler(handler1)

    # load data
    features = load_features(args.scale, args.extractor)

    # load weight lists
    fold_list = args.fold
    print(f'fold:{args.fold}')
    for fold in fold_list:
        if args.version == 'old':
            weight_path = os.path.join(f'/GPUFS/sysu_jhluo_1/wangyh/project/BLCA_TMB/processing/mil classifier',args.description,f'weights/*/fold{fold}/*.pth')
        else:
            weight_path = os.path.join(f'/GPUFS/sysu_jhluo_1/wangyh/data/TCGA_bladder_threshold_80/train/training_details',args.description,f'weights/*/fold{fold}/*.pth')
        weight_lists = glob.glob(weight_path)
        for weight_path in weight_lists:
            pic_folder_root ='/GPUFS/sysu_jhluo_1/wangyh/data/FAH_BLCA/ext_val_dsmil'
            lr, weight, metric,dropout_patch,dropout_node = weight_parser(weight_path,fold)
            
            pic_folder = os.path.join(pic_folder_root,'/'.join(weight_path.split('/')[-5:-1]))
            os.makedirs(pic_folder,exist_ok=True)
            pic_path = os.path.join(pic_folder,weight_path.split('/')[-1].replace('pth','png'))
            
            milnet = load_model(args, weight_path)
            milnet.eval()
            with torch.no_grad():
                avg_score, aucs, conf_mat, fpr, tpr, precision, recall, f1 = test(features, milnet, args)
                log.info(f'lr:{lr},weight:{weight},dropout_patch:{dropout_patch},dropout_node:{dropout_node},fold:{fold},metric:{metric},avg_score: {avg_score},auc for TMB-H: {aucs[1]},f1:{f1},tn_fp_fn_tp:{conf_mat.ravel()[0], conf_mat.ravel()[1], conf_mat.ravel()[2], conf_mat.ravel()[3]}')
                
                metrics_visulization(fpr, tpr, aucs[1], precision, recall, conf_mat, pic_path)
                # log.info(f'avg_score: {avg_score}')
                # log.info(f'auc for TMB-H: {aucs[1]}')
                # log.info(f'tn, fp, fn, tp:{conf_mat.ravel()[0],conf_mat.ravel()[1],conf_mat.ravel()[2],conf_mat.ravel()[3]}')


if __name__ == '__main__':
    main()
