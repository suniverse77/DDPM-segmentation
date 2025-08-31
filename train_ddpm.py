import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import json
import os
import gc
import argparse

from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, Original_Classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev

Epoch = 10
feature_dim = 8448

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(args):
    # 1. Extract feature map
    feature_extractor = create_feature_extractor(**args)                        # args에 있는 key, value 쌍에서 value를 변수로 전달
    
    print(f"Preparing the train set for {args['category']}...")                 # 데이터셋 종류 출력

    train_dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(args['model_type'], args['image_size'])
    )
    X = torch.zeros((len(train_dataset), *args['dim'][::-1]), dtype=torch.float)      # [len(dataset), 8448, 256, 256] 크기 텐서 생성 (feature vector 차원 = 8448)
    y = torch.zeros((len(train_dataset), *args['dim'][:-1]), dtype=torch.uint8)       # [len(dataset), 256, 256] 크기 텐서 생성

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

def train(args):
    train_loader = prepare_data(args)               # 최종적으로 flatten된 feature map 저장

    # Ensemble Training
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()

        # 1. Define Classifier
        classifier = Original_Classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])

        classifier.init_weights()
        classifier = classifier.cuda()
        classifier.train()

        # 2. Define Loss function & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        # 3. Start train
        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0

        for epoch in range(Epoch):
            for X_train, y_train in train_loader:
                '''
                param X_data: pixel representations [num_pixels, feature_dim]
                param y_data: pixel labels [num_pixels]
                '''
                X_train, y_train = X_train.to(dev()), y_train.to(dev()).type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_train)
                loss = criterion(y_pred, y_train)
                acc = multi_acc(y_pred, y_train)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth')
        print('save to:', model_path)
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)

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

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        hard_pred, soft_pred = predict_labels(models, x, args)
        gts.append(label.numpy())   # gts shape  : [test data num, 256, 256]
        hard_preds.append(hard_pred.numpy())  # preds shape: [test data num, 256, 256]
        soft_preds.append(soft_pred.numpy())

    save_predictions(args, dataset.image_paths, hard_preds)
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
    models = load_ensemble(opts, dim=feature_dim, device='cuda')
    evaluation(opts, models)
