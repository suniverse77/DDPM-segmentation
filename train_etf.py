import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import json
import os
import gc
import argparse

from src.utils import setup_seed, multi_acc
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.etf_classifier import compute_iou, predict_labels, save_predictions, save_model, load_model, Pixel_Classifier, ETF_Regularizer
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev

Epoch = 10
alpha = 1               # weight of classifier
beta = 0.4              # weight of regularizer
feature_dim = 8448      # channel dim of feature map

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)                        # args에 있는 key, value 쌍에서 value를 변수로 전달
    
    print(f"Preparing the train set for {args['category']}...")

    train_dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],                                     # 학습에 사용할 이미지 개수 (최대 40개?)
        transform=make_transform(args['model_type'], args['image_size'])
    )

    X = torch.zeros((len(train_dataset), *args['dim'][::-1]), dtype=torch.float)      # [len(dataset), 8448, 256, 256]
    y = torch.zeros((len(train_dataset), *args['dim'][:-1]), dtype=torch.uint8)       # [len(dataset), 256, 256] 

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=dev())
    else:
        noise = None

    for row, (img, label) in enumerate(tqdm(train_dataset)):
        img = img[None].to(dev())

        # U-Net 디코더에서 feature map 추출
        features = feature_extractor(img, noise=noise)
        
        # 해당 클래스가 이미지에서 차지하는 비율이 작으면 라벨 삭제
        for target in range(args['number_class']):
            if target == args['ignore_label']:
                continue
            if 0 < (label == target).sum() < 20:
                label[label == target] = args['ignore_label']

        # 추출한 feature map들을 upsample and concat
        X[row] = collect_features(args, features).cpu()
        y[row] = label

    # Flatten된 결과
    X_train = X.permute(1,0,2,3).reshape(feature_dim, -1).permute(1, 0)
    y_train = y.flatten()

    X_train = X_train[y_train != args['ignore_label']]
    y_train = y_train[y_train != args['ignore_label']]

    train_data = FeatureDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    print(f"***** Number of used label: {args['number_class']}")
    print(f"***** X feature size      : {X_train.shape}")
    print(f"***** y feature size      : {y_train.shape}")

    return train_loader

def train(args, etf_path):
    train_loader = prepare_data(args)

    # Ensemble Training
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()

        # 1. Define Classifier
        classifier = Pixel_Classifier(numpy_class=(args['number_class']), dim=feature_dim)
        regularizer = ETF_Regularizer(numpy_class=(args['number_class']), dim=feature_dim, etf_path=etf_path)

        classifier.init_weights()
        regularizer.init_weights()

        classifier = classifier.cuda()
        regularizer = regularizer.cuda()
        
        classifier.train()
        regularizer.train()

        # 2. Define Loss function & Optimizer
        ce_loss = nn.CrossEntropyLoss()
        # dice_loss = 

        optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001)
        optimizer_regularizer = optim.Adam(regularizer.parameters(), lr=0.001)

        # 3. Start train
        iteration = 0

        for epoch in range(Epoch):
            # Training
            for X_train, y_train in train_loader:
                ## 1. GPU 사용
                X_train, y_train = X_train.to(dev()), y_train.to(dev()).type(torch.long)

                ## 3. Classifier
                optimizer_classifier.zero_grad()
                y_pred = classifier(X_train)
                loss_classifier = ce_loss(y_pred, y_train)

                ## 4. ETF Regularizer
                optimizer_regularizer.zero_grad()
                out_features = regularizer(X_train, y_train)
                loss_regularizer = ce_loss(out_features, y_train)

                ## 4. Gradient update
                loss = alpha * loss_classifier + beta * loss_regularizer

                loss.backward()
                optimizer_classifier.step()
                optimizer_regularizer.step()

                acc = multi_acc(y_pred, y_train)

                iteration += 1
                if iteration % 100 == 0:
                    print(f'Epoch: {epoch}, Iteration: {iteration}')
                    print(f' Loss classifier: {loss_classifier.item():.4f}, Loss regualrizer: {loss_regularizer.item():.4f}')
                    print(f' Total loss: {loss} | Accuracy: {acc}\n')

        save_model(args, classifier, MODEL_NUMBER, iteration, loss, acc)

        MODEL_NUMBER += 1

def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(args['model_type'], args['image_size'])
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=dev())
    else:
        noise = None

    hard_preds, soft_preds, gts = [], [], []
    for img, label in tqdm(dataset):        
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)
        x = features.view(feature_dim, -1).permute(1, 0)

        hard_pred, soft_pred = predict_labels(models, x, args)
        gts.append(label.numpy())   # gts shape  : [test data num, 256, 256]
        hard_preds.append(hard_pred.numpy())  # preds shape: [test data num, 256, 256]
        soft_preds.append(soft_pred.numpy())

    save_predictions(args, dataset.image_paths, soft_preds)
    hard_mIOU = compute_iou(args, hard_preds, gts)
    soft_mIOU = compute_iou(args, soft_preds, gts)

    print(hard_mIOU)
    print(soft_mIOU)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()                                      # 명령줄 인자를 받기 위한 객체 생성
    add_dict_to_argparser(parser, model_and_diffusion_defaults())           # Diffusion 관련 설정 추가

    # Add command argument
    parser.add_argument('--exp', type=str)                                  # json 파일 경로
    parser.add_argument('--name', type=str)                                 # 실험 이름
    parser.add_argument('--seed', type=int, default=0)                      # 랜덤 시드 설정

    args = parser.parse_args()                                              # 명령줄 인자를 해석(파싱)
    setup_seed(args.seed)                                                   # 랜덤 시드 설정하는 함수

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))                                   # json 파일에 있는 값을 딕셔너리로 저장
    opts.update(vars(args))                                                 # 명령줄 인자로 받은 값들을 opts 딕셔너리에 추가 (MODEL_FLAGS, exp, seed)
    opts['image_size'] = opts['dim'][0]
    opts['exp_dir'] = os.path.join(opts['exp_dir'], opts['name'])           # 실험 결과를 저장할 경로 설정

    # Make the directory
    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)                                        # 실험 결과를 저장할 경로 생성      
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))                     # 터미널 명령어 'cp <source> <destination>' 실행 → args.exp 파일을 exp_dir 경로에 복사

    etf_path=os.path.join(opts['exp_dir'], 'etf_matrix.pt')

    print(f"\n*** Experiment folder: {path} ***\n")

    # Train
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) for i in range(opts['model_num'])]

    if not all(pretrained):
        opts['start_model_num'] = sum(pretrained)
        train(opts, etf_path)
    
    # Evaluation
    models = load_model(opts, dim=feature_dim, device='cuda')
    evaluation(opts, models)
