import os
import glob
import shutil
import argparse
import numpy as np
from multiprocessing import Process
from sklearn.model_selection import train_test_split

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def mv_file(pkl_path_list, target_dir):
    for pkl_path in pkl_path_list:
        shutil.move(pkl_path, target_dir)


if __name__ == "__main__":
    """Train the model."""
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/liuzhiwei2/DIABert_train_data/origin/")
    parser.add_argument("--target_dir", default="/liuzhiwei2/DIABert_train_data/train_data")
    parser.add_argument("--ncores", default=20)
    args = parser.parse_args()
    
    # 解析目录
    if ';' in args.source_dir:
        total_train_path = args.source_dir.split(';')

        train_file_list = []
        for train_part_path in total_train_path:
            for file_name in file_list:
                train_part_file_list = glob.glob(f'{train_part_path}/*.pkl')

                if len(train_part_file_list) > 0:
                    train_file_list.extend(train_part_file_list)
    else:
        train_file_list = glob.glob(f'{args.source_dir}/*.pkl')

    train_file_list = [f for f in train_file_list if not is_file_empty(f)]

    # 切分训练集和验证集
    train_list, val_list = train_test_split(train_file_list, test_size=0.1, random_state=42)

    # 移动训练集
    processes = []
    
    ncores = min(int(args.ncores), len(train_list))
    train_dir = os.path.join(args.target_dir, 'train_data')
    
    sublength = int(len(train_list) / ncores)
    for i in range(0, len(train_list), sublength):
        process = Process(target=mv_file, 
                          args=(train_list[i: (i+sublength)],                  
                                train_dir))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
        
    print('train: pkl mv finish!!!')
        
    # 移动验证集
    processes = []
    ncores = min(int(args.ncores), len(val_list))
    val_dir = os.path.join(args.target_dir, 'val_data')
    
    sublength = int(len(val_list) / ncores)
    for i in range(0, len(val_list), sublength):
        process = Process(target=mv_file, 
                          args=(val_list[i: (i+sublength)],                  
                                val_dir))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

    print('val: pkl mv finish!!!')
