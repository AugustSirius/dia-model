from __future__ import annotations

import argparse
import datetime
import logging
import os
import yaml
import random
import numpy as np
import pandas as pd
import math

import lightning.pytorch as ptl
from typing import Any
from sklearn import metrics
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import IterableDataset

import torch
from torch import nn
import torch.utils.data as pt_data
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from model import DIArtModel
from dataset import create_iterable_dataset, Dataset
from utils import set_seeds, mkdir_p

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 设置乘法的精度(fast matrix multiplication algorithms )
torch.set_float32_matmul_precision = 'high'


class Evalute(ptl.LightningModule):
    """evaluate for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: DIArtModel,
            model_name: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.model_name = model_name
        self._reset_metrics()

        # 损失函数
        self.loss_fn = nn.BCELoss()

    def test_step(
            self,
            batch: tuple[Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """Single test step."""
        try:
            # dataloader
            rsm, frag_info, feat, label, file_name, precursor_id = batch
        except:
            # iterable_dataset
            batch = next(iter(batch))
            rsm, frag_info, feat, label, file_name, precursor_id = batch

        rsm = rsm.to(self.device).to(torch.float16)
        frag_info = frag_info.to(self.device).to(torch.float16)
        feat = feat.to(self.device).to(torch.float16)
        label = label.to(self.device).to(torch.float16)

        # preds： (batch , score)
        # truth： (batch , 0/1)
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred = DIArtModel.pred_f16(self.model, rsm, frag_info, feat)
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()

        self.precursor_id_list.extend(precursor_id)
        self.file_name_list.extend(file_name)
        self.pred_list.extend(pred)
        self.label_list.extend(label)

    def on_test_end(self) -> None:
        df = pd.DataFrame({"transition_group_id": self.precursor_id_list,
                           "score": self.pred_list,
                           "label": self.label_list,
                           "file_name": self.file_name_list})
        df.to_csv(os.path.join(self.config['out_path'], f"all_{self.model_name}.csv"), mode='a', header=False, index=None)
        

    def _reset_metrics(self) -> None:
        self.precursor_id_list = []
        self.file_name_list = []
        self.pred_list = []
        self.label_list = []


def main() -> None:
    """Train the model."""
    logging.info("Initializing evaluate.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./eval.yaml")

    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)
    logging.info(f"config:  {args.config}")

    # 设置全局seed
    set_seeds(config['seed'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    if os.path.isdir(config['model_path']):
        ls_model_file = os.popen(fr"ls {config['model_path']}").read()

        # 加载checkpoint的评测
        model_path_list = list([os.path.join(config['model_path'], f) for f in ls_model_file.split('\n') if f.endswith('.ckpt')])

        if config['model_type'] == 'descend':
            # 降序评测
            model_step_id = sorted(range(len(model_path_list)), key=lambda x: int(
                model_path_list[x].split('/')[-1].split('.ckpt')[0].split('=')[-1] if 'step' in model_path_list[x] else 1e4), reverse=True)
        elif config['model_type'] == 'ascend':
            # 升序评测
            model_step_id = sorted(range(len(model_path_list)), key=lambda x: int(
                model_path_list[x].split('/')[-1].split('.ckpt')[0].split('=')[-1] if 'step' in model_path_list[x] else 1e4))

        # 每1个epoch评测一次
        model_path_list = [model_path_list[i] for i in model_step_id]

    else:
        # ckpt od pkl
        model_path_list = [config['model_path']]

    logging.info(f"load model: {';'.join(model_path_list)}")
    mkdir_p(config["out_path"])

    dl = create_iterable_dataset(config['val_data_path'], logging, config, parse='val')
    logging.info(f"load dl: {len(dl)}")

    strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    trainer = ptl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=strategy,
    )

    # 仅加载最新的模型来评测
    for model_path in model_path_list:
        model_name = model_path.split('/')[-1].split('.')[0].replace('=', '')
        logging.info(f"model_name:  {model_name}, device: {device}")

        model = DIArtModel.load_f16_model(model_path)
        model.to(device)
        
        # 输出文件如果存在，则删除
        # out_file = os.path.join(config['out_path'], f"all_{config['eval_data_type']}_{model_name}.csv")
        # if os.path.exists(out_file):
        #     os.remove(out_file)

        evaluate = Evalute(config, model, model_name)
        trainer.test(evaluate, dataloaders=dl)
        
        

if __name__ == "__main__":
    main()
    
# cd /wangshuaiyao/DIArt_model_250711/DIArt_model_250711; python eval.py --config /wangshuaiyao/DIArt_model_250711/DIArt_model_250711/yaml/eval.yaml

# DIArt_model_250711/DIArt_model_250711/eval.py