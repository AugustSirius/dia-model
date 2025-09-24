import os
import os.path
import random
random.seed(123)

import logging
from optparse import OptionParser
import numpy as np
import pandas as pd
import glob
import math

import torch.utils.data as pt_data
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from random import sample
from multiprocessing import Process

logging.basicConfig(level=logging.INFO, format="DIArt: %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.info("Welcome to DIArt!")


def mkdir_p(dirs):
    """
    make a directory (dir) if it doesn't exist
    """
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    return True, 'OK'

class Dataset(pt_data.Dataset):
    def __init__(self):
        self.rsm = None
        self.frag_info = None
        self.feat = None
        self.label = None
        self.file = None
        self.precursor_id = None

    def __getitem__(self, idx):
        return_dict = {"rsm": self.rsm[idx],
                       "frag_info": self.frag_info[idx],
                       "feat": self.feat[idx],
                       "label": self.label[idx],
                       "file": self.file[idx],
                       "precursor_id": self.precursor_id[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self):
        pass

class FeatureEngineer():
    def __init__(
            self,
            max_length: int = 30,
            max_mz: int = 1801,
            max_charge: int = 10,
            max_irt: int = 600,
            max_nr_peaks: int = 20,
            max_rt: int = 6400,
            max_delta_rt: int = 1000,
            max_intensity: int = 10000,
    ) -> None:
        self.max_length = max_length
        self.max_mz = max_mz
        self.max_charge = max_charge

        self.max_irt = max_irt
        self.max_nr_peaks = max_nr_peaks
        self.max_rt = max_rt
        self.max_delta_rt = max_delta_rt
        self.max_intensity = max_intensity

    @staticmethod
    def process_intensity(intensity):
        if np.sum(intensity) < 1e-6:
            return intensity

        rsm_max = np.amax(intensity, (1, 2, 3))

        intensity = pow(intensity, 1/3)
        rsm_max = pow(rsm_max, 1/3)

        # process_feat
        rsm_max_new = rsm_max.copy()
        rsm_max_new[rsm_max_new < 1e-6] = 1
        return np.divide(intensity, rsm_max_new.reshape(rsm_max_new.shape[0], 1, 1, 1)), rsm_max / pow(1e10, 1/3)

    def process_feat(self, feat):
        # sequence_length, precursor_mz, charge, precursor_irt,
        # nr_peaks, assay_rt_kept, delta_rt_kept, max_intensity
        scale_factor = [self.max_length, self.max_mz, self.max_charge, self.max_irt,
                        self.max_nr_peaks, self.max_rt, self.max_delta_rt, 1]
        # sequence_length
        feat = feat / scale_factor
        return feat

    def process_frag_info(self, frag_info):
        frag_info[:, :, 0] = frag_info[:, :, 0] / self.max_mz
        frag_info[:, :, 1] = frag_info[:, :, 1] / self.max_intensity
        return frag_info


def gen_train_data(options):
    final_res = glob.glob('%s/*/*.pkl' % options.pkl_dir)
    fdr_path_list = list(set(['/'.join(v.split('/')[:-1]) for v in final_res if (v.split('/')[-2] not in ['', 'decoy_matrix', 'diann17_decoy', 'diann17_decoy_rt_shift', 'diann18_target', 'diann18_target_rt_shift'])]))
    logger.info('total: %s, %s' % (len(fdr_path_list), fdr_path_list[0]))
    
    # filter finish
    finish_file_list = glob.glob('%s/%s/*csv' % (options.pkl_dir, options.task_name))
    finish_file_list = [f.split('/')[-1].replace(f'_{options.task_name}_feature.csv', '') for f in finish_file_list]
    fdr_path_list = [f for f in fdr_path_list if f.split('/')[-1] not in finish_file_list]
    print('after filter, curr: %s' % len(fdr_path_list))

    fdr_filename_list = [item.split("/")[-1] for item in fdr_path_list]
    logger.info("fdr_filename len: %s, %s" % (len(fdr_filename_list), fdr_filename_list[:1]))  

    fdr_dir = os.path.join(options.pkl_dir, options.task_name)
    mkdir_p(fdr_dir)

    # 加载梯度和仪器信息
    info_file = pd.read_csv(options.info_path)
    
    RT = [26, 44, 60, 75, 90, 105, 120, 30, 35, 40, 50, 55, 65, 70, 80, 85, 95, 100, 110, 115] + list(range(125,241,5))
    INSTRUMENT = ['Orbitrap exactive hf',
                  'Orbitrap exactive hf-x',
                  'Orbitrap exploris 480',
                  'Orbitrap fusion lumos',
                  'Tripletof 5600',
                  'Tripletof 6600',
                  'Other']
    rt_s2i = {v: i for i, v in enumerate(RT)}
    instrument_s2i = {v: i for i, v in enumerate(INSTRUMENT)}

    info_file['rt_min'] = np.where(info_file['rt_min'].isin({45, 47}), 44, info_file['rt_min'])
    info_file['rt_min'] = info_file['rt_min'].apply(lambda x: rt_s2i[x])
    info_file['Instruments'] = info_file['Instruments'].apply(lambda x: instrument_s2i[x])
    info_file['info'] = info_file[['rt_min', 'Instruments']].apply(tuple, axis=1)
    info_dict = dict(zip(info_file['mzml_name'], info_file['info']))

    feature_list = ['sequence_length', 'precursor_mz', 'charge', 'precursor_irt', 'nr_peaks', 
                    'assay_rt_kept', 'delta_rt_kept', 'max_intensity', 'rt', 'instrument']

    ncores = min(options.ncores, len(fdr_filename_list))
    if ncores > 1:
        processes = []
        sublength = int(len(fdr_path_list) / ncores) 
        for i in range(0, len(fdr_path_list), sublength):
            process = Process(target=construct_data_set, args=(fdr_path_list[i:(i+sublength)],
                              fdr_filename_list[i:(i+sublength)],
                              fdr_dir,
                              feature_list,
                              info_dict,
                              options.task_name,
                              options.batch_size))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
    else:
        construct_data_set(fdr_path_list,
                           fdr_filename_list,
                           fdr_dir,
                           feature_list,
                           info_dict,
                           options.task_name,
                           options.batch_size)



def construct_data_set(data_path_list,
                       filename_list,
                       feat_dir,
                       feature_list,
                       info_dict,
                       task_name,
                       batch_size=2048):
    feature_engineer = FeatureEngineer()

    for i in range(len(data_path_list)):
        try:
            pkl_list = [i for i in os.listdir(data_path_list[i]) if i.endswith("pkl")]
            file_precursor_id, file_rsm, file_frag_info = [], [], []
            file_precursor_feat, file_target = [], []
            logger.info('pkl_num: %s' % (len(pkl_list)))

            for file in pkl_list:
                chrom_file = data_path_list[i] + '/' + file

                f = open(chrom_file, "rb")
                precursor_data = pickle.load(f)
                f.close()

                # precursor_id: (b, 2)
                # precursor_feat: (b, 8)
                # rsm: (b, 8, 72, 16)
                # frag_info: (b, 72, 4)
                # score: (b)
                precursor, precursor_feat, rsm, frag_info = precursor_data
                precursor = np.array(precursor)
                precursor_id = precursor[:, 0].tolist()

                file_precursor_id.extend(precursor_id)  # precursor_id
                file_rsm.append(rsm)
                file_frag_info.append(frag_info)
                file_precursor_feat.append(precursor_feat)
                file_target.extend([0 if p.startswith("DECOY") else 1 for p in precursor_id])

            # save xrm_feature
            file_rsm = np.concatenate(file_rsm, axis=0)
            file_frag_info = np.concatenate(file_frag_info, axis=0)
            file_precursor_feat = np.concatenate(file_precursor_feat, axis=0)
            logger.info('file: %s, precursor num: %s' % (filename_list[i], len(file_precursor_id)))
            logger.info(f'precursor_feat: {file_precursor_feat.shape}')

            # feature engineer
            file_rsm, rsm_max = FeatureEngineer.process_intensity(file_rsm)
            logger.info(f'rsm max(rsm) : {np.max(file_rsm)}, min(rsm): {np.min(file_rsm)}')       
            file_frag_info = feature_engineer.process_frag_info(file_frag_info)
            logger.info(f'frag_info max(frag_info) : {np.max(file_frag_info)}, min(frag_info): {np.min(file_frag_info)}') 

            # 拼接rsm_max
            file_precursor_feat = np.column_stack((file_precursor_feat[:, :7], rsm_max))
            file_precursor_feat = feature_engineer.process_feat(file_precursor_feat)
            logger.info(f'frag_info max(precursor_feat) : {np.max(file_precursor_feat)}, min(precursor_feat): {np.min(file_precursor_feat)}')

            file_rsm = file_rsm.swapaxes(1, 2)
            logger.info(f'rsm shape: {file_rsm.shape}')
            logger.info(f'frag_info: {file_frag_info.shape}')
            logger.info(f'precursor_feat: {file_precursor_feat.shape}')

            # 增加仪器和梯度信息，合并feat
            pr_ids = len(file_precursor_id)
            try:
                rt, instrument = info_dict[filename_list[i] + '.mzML']
            except:
                # astral的梯度为24min
                rt = 0
                # 仪器类型为atral时，默认设置为other
                instrument = 6
            rt_np = np.array([rt]).repeat(pr_ids).reshape(-1, 1)
            instrument_np = np.array([instrument]).repeat(pr_ids).reshape(-1, 1)
            file_precursor_feat = np.concatenate((file_precursor_feat, rt_np, instrument_np), axis=1)
            logger.info(f'add rt={rt} and instrument={instrument}, precursor_feat: {file_precursor_feat.shape}')

            # gen df
            df = pd.DataFrame(file_precursor_feat, index=file_precursor_id, columns=feature_list)
            df["target"] = file_target
            df["filename"] = filename_list[i].replace('/', '')

            if feat_dir is not None:
                file_dir = os.path.join(feat_dir, f"{filename_list[i]}_{task_name}_feature.csv")
                df.to_csv(file_dir, sep="\t")
                logger.info('feature save: %s' % file_dir)

            decoy_num = len(df[df['target'] == 0])
            target_num = len(df[df['target'] == 1])
            logger.info("decoy: {}, target: {}, total: {}".format(decoy_num, target_num, len(df)))
            assert decoy_num + target_num == len(df)
            
            if len(df) == 0:
                continue

            # 缓存
            data_set = Dataset()
            data_set.rsm = file_rsm
            data_set.feat = file_precursor_feat
            data_set.frag_info = file_frag_info
            data_set.label = file_target
            data_set.precursor_id = file_precursor_id
            data_set.file = [filename_list[i] for _ in range(len(file_precursor_id))]

            logger.info(f'{data_path_list[i]} save to {feat_dir} , len: {len(data_set)}')
            
            task = defaultdict(list)
            pkl_num = math.ceil(len(data_set) / batch_size)
            for k in range(len(data_set)):
                file_pkl_dir =  os.path.join(feat_dir, f'{filename_list[i]}_{k % pkl_num}_{task_name}.pkl')
                task[file_pkl_dir].append(data_set[k])

            for pkl_name in task:
                output_pkl = open(pkl_name, "wb")
                output_pkl.write(pickle.dumps(task[pkl_name], protocol=4))
                output_pkl.close()
        except Exception as e:
            logger.info('*********************load %s error: %s*********************' % (data_path_list[i], e))
            continue


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--pkl_dir", type="string", default="/liuzhiwei2/DIArt2/data/matrix/mkxrm1/train_620_diann18_rt_add_one",
                      help=".pkl directory, better in /mnt")
    parser.add_option("--task_name", type="string", default="diann18_target", help="task_name")
    parser.add_option("--seed", type="int", default="123", help="random seed for decoy generation.")
    parser.add_option("--batch_size", type="int", default=2048)
    parser.add_option("--ncores", type="int", default=20, help="number of CPU cores")
    parser.add_option("--info_path", default="/liuzhiwei2/DIArt_model_250623/rt_instrument_info_240509.csv")

    (options, args) = parser.parse_args()
    logger.info('getdata begin!!!, batch_size: %s' % (options.batch_size))
    gen_train_data(options)
    logger.info('getdata end!!!!')
