import argparse
import os
from importlib import import_module
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from util import read_json


def load_model(saved_model, config, dataset_name, device):
    model_cls = getattr(import_module("model"), args.model)
    num_classes = getattr(import_module("dataset"), dataset_name).num_classes
    
    config.pretrained = False
    model = model_cls(
        config,
        num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, args.config, args.dataset, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


@torch.no_grad()
def TTA(data_dir, model_dir, output_dir, args, time=5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, args.config, device).to(device)
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    augmentation_module = getattr(import_module("dataset"), args.valid_augmentation)
    transform = augmentation_module(resize=[224, 224], mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")

    preds_per_time = []

    for i in range(time):
        preds = torch.empty((0), dtype=torch.float32, device='cuda')
        with torch.no_grad():
            model.eval()
            for idx, images in enumerate(tqdm(loader, mininterval=0.5, ncols=100)):
                images = images.to(device)
                pred = model(images)
                preds = torch.cat((preds, pred), dim=0)
            preds_per_time.append(preds.cpu())

    all_logits = torch.stack(preds_per_time, dim=0)  # time, 12600, 18
    summation = torch.sum(all_logits, dim=0)  # 12600, 18
    TTA_result = torch.argmax(summation, dim=-1)

    info['ans'] = TTA_result
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'TTA Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'), help="사용할 best model 이 담긴 디렉토리 경로. 지정 필수")
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    
    configs = read_json(f"{model_dir}/config.json")
    setattr(args, "config", configs["config"]) # 학습 시 사용했던 model의 configuration
    setattr(args, "model", configs["model"]) # 학습 시 사용했던 모델의 이름을 읽어옴
    setattr(args, "valid_augmentation", configs["valid_augmentation"]) # 학습 시 valid set에 적용했던 augmentation
    setattr(args, "dataset", configs["dataset"]) 
    
    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
