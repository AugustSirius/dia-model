import torch
import torch.nn as nn
import math
import numpy as np
import torch.utils.data as pt_data

import warnings

warnings.filterwarnings("ignore")

# 设置乘法的精度(fast matrix multiplication algorithms )
torch.set_float32_matmul_precision = 'high'


class Dataset(pt_data.Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.rsm = None
        self.file = None
        self.label = None
        self.precursor_id = None
        self.feat = None
        self.frag_info = None

    def __getitem__(self, idx):
        return_dict = {"file": self.file[idx],
                       "rsm": self.rsm[idx],
                       "label": self.label[idx],
                       "precursor_id": self.precursor_id[idx],
                       "feat": self.feat[idx],
                       "frag_info": self.frag_info[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self, scaler_list, embedding=False):
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
        
        RT = [26, 44, 60, 75, 90, 105, 120, 30, 35, 40, 50, 55, 65, 70, 80, 85, 95, 100, 110, 115] + list(range(125,241,5))
        self.rt_s2i = {v: i for i, v in enumerate(RT)}
        
        INSTRUMENT = ['Orbitrap exactive hf',
                      'Orbitrap exactive hf-x',
                      'Orbitrap exploris 480',
                      'Orbitrap fusion lumos',
                      'Tripletof 5600',
                      'Tripletof 6600',
                      'Other']
        self.instrument_s2i = {v: i for i, v in enumerate(INSTRUMENT)}
        
    def get_rt_s2i(self, rt):
        return self.rt_s2i(rt)
        
    def get_instrument_s2i(instrument):
        return self.instrument_s2i(instrument)

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


class NLinearMemoryEfficient(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NumEmbeddings(nn.Module):
    def __init__(
            self,
            n_features: int,
            d_embedding: int,
            embedding_arch: list,
            d_feature: int,
    ) -> None:
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'shared_linear',
            'relu',
            'layernorm',
            'batchnorm',
        }

        # NLinear_ =  NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            assert d_embedding is not None
            layers.append(
                NLinearMemoryEfficient(n_features, d_feature, d_embedding)
            )
        elif embedding_arch[0] == 'shared_linear':
            layers.append(
                nn.Linear(d_feature, d_embedding)
            )
        d_current = d_embedding

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinearMemoryEfficient(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else nn.LayerNorm([n_features, d_current])
                if x == 'layernorm'
                else nn.BatchNorm1d(n_features)
                if x == 'batchnorm'
                else nn.Identity()
            )
            if x in ['linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x) 
        return output * self.weight


class ConvLayer(nn.Module):
    def __init__(self,
                 embedding_len,
                 n_rsm_features,
                 n_frags=72,
                 in_channels=1,
                 mid_channels=4,
                 out_channels=16,
                 kernel_size=(1, 3, 3),
                 padding=(0, 1, 1),
                 eps=1e-5):
        super(ConvLayer, self).__init__()
        # 一维卷积改为三维卷积，每个fragment的8行数据对应8个channel
        # 第一个卷积层输入1个channel,输出4个channel
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=mid_channels,
                               kernel_size=kernel_size,
                               padding=padding)

        # layer norm
        self.norm1 = nn.LayerNorm([mid_channels, n_frags, embedding_len, n_rsm_features], eps=eps)
#         self.norm1 = RMSNorm([mid_channels, n_frags, embedding_len, n_rsm_features], eps=eps)
        
        # 第二个卷积层输入4个channel,输出16个channel
        self.conv2 = nn.Conv3d(in_channels=mid_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        # layer norm
        self.norm2 = nn.LayerNorm([out_channels, n_frags, embedding_len // 2, n_rsm_features // 2], eps=eps)
#         self.norm2 = RMSNorm([out_channels, n_frags, embedding_len // 2, n_rsm_features // 2], eps=eps)

        # pool
        self.pool_0 = nn.AvgPool3d((1, 2, 2))
        self.pool_1 = nn.AvgPool3d((1, 2, 4))
        self.activation = nn.ReLU()

    # @torch.compile
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)  # (batch, 1, 72, 8, 16) -> (batch, 4, 72, 8, 16)
        x = self.activation(x)
        x = self.norm1(x)  # (batch, 4, 72, 8, 16) -> (batch, 4, 72, 8, 16)
        
        x = self.pool_0(x)  # (batch, 4, 72, 8, 16) -> (batch, 4, 72, 4, 8)
        x = self.conv2(x)  # (batch, 4, 72, 4, 8) -> (batch, 16, 72, 4, 8)
        x = self.activation(x)
        x = self.norm2(x)  # (batch, 16, 72, 4, 8) -> (batch, 16, 72, 4, 8)
        x = self.pool_1(x)  # (batch, 16, 72, 4, 8) -> (batch, 16, 72, 2, 2)

        x = x.transpose(1, 2).contiguous().view(-1, 72, 64)
        # (batch, 16, 72, 2, 2) -> (batch, 72, 16, 2, 2) -> (batch, 72, 64)
        return x


class DIArtModel(nn.Module):
    def __init__(self, 
                 n_num_features=8,
                 n_rsm_features=16,
                 embedding_len=8,
                 channels=72,
                 dropout=0.2,
                 embedding_dim=8,
                 eps=1e-5):
        """
        Args:
            n_num_features: 数值型特征的数量
            n_rsm_features：rsm矩阵在rt上的维度
            embedding_len： rsm矩阵中，每个fragment的维度
            channels: rsm矩阵中，fragment的数量
            embedding_dim：梯度、质谱仪型号等类别特征Embedding的维度
            eps: norm的eps

        """
        super(DIArtModel, self).__init__()

        self.n_num_features = n_num_features
        self.n_rsm_features = n_rsm_features
        self.embedding_len = embedding_len
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.eps = eps

        # embedding_len作为channels
        self.conv = ConvLayer(self.embedding_len, self.n_rsm_features, in_channels=1, mid_channels=4, out_channels=16, eps=self.eps)

        # transformer, 在rt维度上
        self.transformer_layer_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64*4,
                                                              batch_first=True, layer_norm_eps=self.eps)
        self.transformer_encoder_1 = nn.TransformerEncoder(self.transformer_layer_1, num_layers=4)

        # transformer, 在fragemnt维度上
        self.transformer_layer_2 = nn.TransformerEncoderLayer(d_model=72, nhead=8, dim_feedforward=72*4,
                                                              batch_first=True, layer_norm_eps=self.eps)
        self.transformer_encoder_2 = nn.TransformerEncoder(self.transformer_layer_2, num_layers=4)

        # 数值型变量embedding
        embedding_arch = ['shared_linear', 'batchnorm', 'relu']
        # n_features: embedding维度；d_feature：输入维度；d_embedding：输出维度

        # self.num_embeddings_1 = NumEmbeddings(n_features=channels, d_embedding=16,
        #                                       embedding_arch=embedding_arch,
        #                                       d_feature=4)
        
        self.num_embeddings_1 = NumEmbeddings(n_features=channels, d_embedding=16,
                                              embedding_arch=embedding_arch,
                                              d_feature=3)

        self.num_embeddings_2 = NumEmbeddings(n_features=1, d_embedding=64,
                                              embedding_arch=embedding_arch,
                                              d_feature=self.n_num_features)

        self.linear_0 = nn.Linear(72, 128)
        self.linear_1 = nn.Linear(80, 32)
        
        # 类别型变量embedding (20240326适配synthedia的数据)
        self.rt_embedding = nn.Embedding(44, self.embedding_dim)
        self.instructment_embedding = nn.Embedding(7, self.embedding_dim)

        # 全连接层
        self.linear_2 = nn.Linear(208, 256)
        self.linear_3 = nn.Linear(256, 64)
        self.linear_out = nn.Linear(64, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        self.apply(init_weights)

    def forward(self, rsm, frag_info, feats):
        """
        Args:
        rsm 维度: (batch_size, 72, 8, 16)
        frag_info 维度: (batch_size, 72, 4)
        feats 维度: (batch_size, 10)

        """
        # rsm (batch, 72, 8, 16)，对应于图像处理的BCHW
        ########################################################################################################
        # conv + LayerNorm
        rsm = self.dropout(self.conv(rsm))  # (batch, 72, 8, 16) ==> (batch, 72, 64)

        # rt维度transformer encoder
        rsm = self.transformer_encoder_1(rsm)  # (batch, 72, 64)

        # 计算frag_info数据
        # frag_info 从(batch, 72, 4) ==> (batch, 72, 16)
        frag_info = self.num_embeddings_1(frag_info)

        # 合并矩阵(batch, 72, 64) +  (batch, 72, 16)==> (batch, 72, 80) (batch, frag, rt_feat)
        rsm = torch.cat((rsm, frag_info), dim=-1)

        # fragment维度transformer
        rsm = rsm.transpose(1, 2)  # (batch, 72, 80) ==> (batch, 80, 72)(batch, rt_feat, frag)
        rsm = self.transformer_encoder_2(rsm)  # (batch, 80, 72)

        # 聚合不同fragment的信息
        rsm = self.dropout(self.activation(self.linear_0(rsm)))  # (batch, 80, 72) ==> (batch, 80, 128)
        rsm = rsm.transpose(1, 2)  # (batch, 80, 128) ==> (batch, 128, 80)

        # fragment维度上，聚合不同rt的信息
        rsm = self.dropout(self.activation(self.linear_1(rsm)))  # (batch, 128, 80) ==> (batch, 128, 32)
        
        dense_rep = torch.mean(rsm, dim=-1)  # (batch, 128)

        # 处理数值型特征
        precursor_feats = self.num_embeddings_2(feats[:, :8].unsqueeze(dim=-2)).squeeze(dim=-2)  # (batch, 8) ==> (batch, 64)
        
        # 处理梯度和仪器等类别特征
        rt = self.rt_embedding(feats[:, 8].long()) # (batch, 1) ==> (batch, 8) 
        instructment = self.instructment_embedding(feats[:, 9].long()) # (batch, 1) ==> (batch, 8) 

        # rsm + feat
        dense_rep = torch.cat((dense_rep, precursor_feats, rt, instructment), dim=1)  
        # (batch, 128) + (batch, 64) + + (batch, 8) + + (batch, 8)  ==> (batch, 208)

        ########################################################################################################
        ###全连接层增加了dropout
        dense_rep = self.dropout(self.activation(self.linear_2(dense_rep)))  # (batch, 208)  ==> (batch, 256)
        dense_rep = self.dropout(self.activation(self.linear_3(dense_rep)))  # (batch, 256)  ==> (batch, 64)

        rep_score = self.linear_out(dense_rep)  # (batch, 64)  ==> (batch, 1)

        rep_score = torch.flatten(rep_score)
        # 裁剪，fillna
        rep_score = torch.clamp(rep_score, min=-1000.0, max=1000.0)
        rep_score = torch.nan_to_num(rep_score)

        return rep_score
    
    @classmethod
    def load(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        diart_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in state_dict.keys():
            state_dict = state_dict['state_dict']
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(diart_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(diart_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            # print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))}")
        diart_model.load_state_dict(model_state, strict=False)
        diart_model.eval()
        return diart_model
    
    @classmethod
    def load_f16_model(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        diart_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in state_dict.keys():
            state_dict = state_dict['state_dict']
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(diart_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(diart_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            # print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))}")
        diart_model.load_state_dict(model_state, strict=False)
        diart_model.half()
        diart_model.eval()
        return diart_model
    
    @classmethod
    def load_f16_pt_model(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        diart_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        
        model_state = state_dict['module']

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(diart_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(diart_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))}")
        diart_model.load_state_dict(model_state, strict=False)
        diart_model.half()
        diart_model.eval()
        return diart_model

    @classmethod
    def load_ema(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        diart_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        if "model_ema" in state_dict.keys():
            state_dict = state_dict['model_ema']
        model_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(diart_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(diart_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            # print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))[:2]}")
        diart_model.load_state_dict(model_state, strict=False)
        return diart_model

    @staticmethod
    def pred(model,
             rsm,
             frag_info,
             feats):
        """model pred."""
        with torch.no_grad():
            score = model(rsm, frag_info, feats)
            sigmod = nn.Sigmoid()
            score = sigmod(score)
        return score

    @staticmethod
    def pred_f16(model,
                 rsm,
                 frag_info,
                 feats):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                score = model(rsm, frag_info, feats)
                sigmod = nn.Sigmoid()
                score = sigmod(score)
        return score

if __name__ == '__main__':
#     rsm = torch.randn(100, 72, 8, 16)
#     frag_info = torch.randn(100, 72, 4)
#     feat = torch.randn(100, 8)
    
#     RT = [26, 44, 60, 75, 90, 105, 120]
#     INSTRUMENT = ['Orbitrap exactive hf',
#                   'Orbitrap exactive hf-x',
#                   'Orbitrap exploris 480',
#                   'Orbitrap fusion lumos',
#                   'Tripletof 5600',
#                   'Tripletof 6600']
#     rt_s2i = {v: i for i, v in enumerate(RT)}
#     instrument_s2i = {v: i for i, v in enumerate(INSTRUMENT)}

#     # 指定梯度和设备
#     rt = 20*[26] + 60*[60] + 20*[105]
#     instrument = 20*['Orbitrap exactive hf'] + 60*['Orbitrap fusion lumos'] + 20*['Tripletof 6600']

#     rt_tensor = torch.tensor([rt_s2i[x] for x in rt]).unsqueeze(dim=-1)
#     instrument_tensor = torch.tensor([instrument_s2i[x] for x in instrument]).unsqueeze(dim=-1)
#     feat = torch.cat((feat, rt_tensor, instrument_tensor), dim=1)


#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
#     rsm = rsm.to(device)
#     frag_info = frag_info.to(device)
#     feat = feat.to(device)
#     print('rsm: ', rsm.shape)
#     print('frag_info: ', frag_info.shape)
#     print('feat: ', feat.shape)

    # 初始化测试
    model = DIArtModel()
    # model.to(device)
    print('DIArt模型规模： {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # with torch.no_grad():
    #     score = model(rsm, frag_info, feat)
    # print('score', score.shape)

