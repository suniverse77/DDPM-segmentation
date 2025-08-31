import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pandas as pd
import math
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask
from src.data_util import get_palette, get_class_names
from PIL import Image


class Pixel_Classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super().__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)

class ETF_Regularizer(nn.Module):
    def __init__(self, numpy_class, dim, etf_path):
        super().__init__()

        self.etf_path = etf_path

        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32)
            )
            d_model = 32
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128)
            )
            d_model = 128

        self.etf = self.generating_etf(d_model, numpy_class)

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        
        return orth_vec

    def generating_etf(self, d_model, n_cls):
        if os.path.exists(self.etf_path):
            etf = torch.load(self.etf_path)
        else:
            orth_vec= self.generate_random_orthogonal_matrix(d_model,n_cls)
            i_nc_nc = torch.eye(n_cls)
            one_nc_nc: torch.Tensor = torch.mul(torch.ones(n_cls, n_cls), (1 / n_cls))
            etf = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc), math.sqrt(n_cls / (n_cls - 1)))
            torch.save(etf, self.etf_path)

        return etf

    def calcul_feature_centure(self, x, y):
        classes = torch.unique(y)
        center_features = []
        center_labels = []

        batch_size = x.shape[0]

        for batch in range(batch_size):                 # 배치마다 계산
            for cls in classes:                         # 현재 배치의 해당 클래스 마스크
                mask = (y[batch] == cls)                # mask size   : [H, W]
                feature = x[batch].permute(1, 2, 0)     # feature size: [H, W, C]
                masked_feature = feature[mask]

                center_feature = masked_feature.mean(dim=0)
                center_features.append(center_feature)
                center_labels.append(cls)

        center_features = torch.stack(center_features)  # center features size: [num_class, reconstruct_dim]
        center_labels = torch.tensor(center_labels)     # center labels size  : [num_class]
        # print(len(classes), center_features.shape, center_labels.shape)

        return center_features, center_labels

    def forward(self, x, y):
        # feature_centure, center_labels = self.calcul_feature_centure(x, y)
        x = self.layers(x)
        etf = self.etf.to(x.device)
        out = x @ etf
        
        return out

def hard_voting(models, features, args):
    seg_mode_ensemble = []

    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            # 1. 
            classifier = models[MODEL_NUMBER]
            preds = classifier(features.cuda())

            y_pred_softmax = torch.log_softmax(preds, dim=1)
            _, img_seg = torch.max(y_pred_softmax, dim=1)
            img_seg = img_seg.reshape(shape=args['dim'][:-1])
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]

    return img_seg

def soft_voting(models, features, args):
    prob = []

    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            classifier = models[MODEL_NUMBER]
            preds = classifier(features.cuda())

            prob.append(torch.log_softmax(preds, dim=1))
        
        probs = torch.stack(prob)
        y_pred = torch.mean(probs, dim=0)
        _, img_seg = torch.max(y_pred, dim=1)

        img_seg = img_seg.reshape(shape=args['dim'][:-1])
        img_seg = img_seg.cpu().detach()

    return img_seg

def weighted_voting(preds, args):
    entropy = Categorical(logits=preds).entropy()   # 각 픽셀의 entropy (값이 클수록 불확실함)
    # weight = F.softmax(-entropy)
    weight = torch.exp(-entropy)

    preds = torch.stack([row * w for row, w in zip(preds, weight)])  # weighted entropy
    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, img_seg = torch.max(y_pred_softmax, dim=1)
    img_seg = img_seg.reshape(shape=args['dim'][:-1])
    img_seg = img_seg.cpu().detach()

    return img_seg

def predict_labels(models, features, args):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    hard_seg = hard_voting(models, features, args)
    soft_seg = soft_voting(models, features, args)

    return hard_seg, soft_seg

def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(os.path.join(args['exp_dir'], 'visualizations', filename + '.jpg'))


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])
    class_ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()
    pixel_counts = Counter()

    # 이미지 한 장씩 가져오기
    for pred, gt in zip(preds, gts):
        for target_num in class_ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)                # 해당 클래스가 존재하는 영역 1로 masking
            gts_tmp = (gt == target_num).astype(int)                    # 해당 클래스가 존재하는 영역 1로 masking
            unions[target_num] += (preds_tmp | gts_tmp).sum()           # 합집합 계산
            intersections[target_num] += (preds_tmp & gts_tmp).sum()    # 교집합 계산

            pixel_counts[target_num] += gts_tmp.sum()                   # 해당 클래스의 픽셀 개수
    
    iou_classes = []

    for target_num in class_ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        iou_classes.append(iou)
    mIOU = np.array(iou_classes).mean()

    save_path = os.path.join(args['exp_dir'], str(args['category']))
    
    return mIOU

    # data = []
    # for target_num in class_ids:
    #     if target_num == args['ignore_label']: 
    #         continue
    #     if print_per_class_ious:
    #         class_name = class_names[target_num]
    #         iou = iou_classes[target_num]
    #         pixel = pixel_counts[target_num]
    #         data.append({"Class": class_name, "IOU": iou, "Pixel": pixel})
    # data.append({"Class": 'mIOU', "IOU": mIOU, "Pixel": '-'})

    # df = pd.DataFrame(data)
    # df.to_csv(save_path, index=False)
    # print(df)


def save_model(args, classifier, MODEL_NUMBER, iteration, loss, acc):
    model_path = os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth')

    print('save to:', model_path)

    torch.save({'model_state_dict': classifier.state_dict()}, model_path)

    with open(os.path.join(args['exp_dir'], 'Record.txt'), 'a') as f:
        f.write(f'{iteration} - Loss: {loss:.5f} | Acc: {acc:.5f}\n')

def load_model(args, dim, device='cpu'):
    models = []

    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']

        classifier = Pixel_Classifier(numpy_class=(args['number_class']), dim=dim)
        classifier.load_state_dict(state_dict)
        classifier = classifier.to(device).eval()

        models.append(classifier)
        
    return models
