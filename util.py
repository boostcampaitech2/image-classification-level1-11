from pathlib import Path
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import torch


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def update_argument(args, configs):
    for arg in configs:
        if arg in args:
            setattr(args, arg, configs[arg])
        else:
            raise ValueError(f"no argument {arg}")
    return args


def ages_subdiv_to_origin(sdage):
    result = []
    for age in sdage:
        if age < 2:
            result.append(0)
        elif age < 5:
            result.append(1)
        else:
            result.append(2)
    return result


def draw_confusion_matrix(true, pred, dir, num_classes):
    cm = confusion_matrix(true, pred)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None], 
                index=list(range(num_classes)), columns=list(range(num_classes)))
    df = df.fillna(0)  # NaN 값을 0으로 변경

    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig(f"{dir}/confusion_matrix.png")
    plt.close('all')


def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']