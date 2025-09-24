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

from typing import Any
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from utils import mkdir_p, get_prophet_result, eval_predict

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def evaluate_fdr(df, config, sw, model_name, epoch, species_pr=None) -> None:    
    print(f"evaluate metrics:")
    # epoch = int(model_name.split('-')[0].replace('epoch', ''))

    df.columns = ['transition_group_id', 'score', 'label', 'file_name']
    df = df.drop_duplicates(subset=['transition_group_id', 'file_name'], keep='first')
    
    pred = df['score'].astype(float).tolist()
    label = df['label'].astype(float).tolist()
    auc, accuracy = eval_predict(pred, label)
    
    # 去掉logit
    loss_fn = nn.BCELoss()
    loss = loss_fn(torch.tensor(pred), torch.tensor(label))
    
    sw.add_scalar("eval/loss", loss.item(), epoch)
    sw.add_scalar("eval/auc", auc, epoch)
    sw.add_scalar("eval/acc", accuracy, epoch)

    print(
        f"epoch={epoch}, loss={loss.item():.5f}, auc: {auc}, acc: {accuracy}"
    )


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

        model_path_list = [model_path_list[i] for i in model_step_id]
    else:
        # ckpt od pkl
        model_path_list = [config['model_path']]

    print('model: ', model_path_list)
    
    # 初始化tensorboard
    mkdir_p(config["tb_summarywriter"])
    config["tb_summarywriter"] = os.path.join(config["tb_summarywriter"], datetime.datetime.now().strftime(
        "diart_eval_%y_%m_%d_%H_%M_%S"
    ))
    sw = SummaryWriter(config["tb_summarywriter"])
    
    epoch = 0
    for model_path in model_path_list:
        model_name = model_path.split('/')[-1].split('.')[0].replace('=', '')
        logging.info(f"model_name:  {model_name}, epoch: {epoch}")
        
        df = pd.read_csv(os.path.join(config['out_path'], f"all_{model_name}.csv"))
        evaluate_fdr(df, config, sw, model_name, epoch)
        epoch += 1

if __name__ == "__main__":
    main()