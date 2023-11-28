import pandas as pd
import torch

data = pd.read_excel('./대화체.xlsx')
data.head()

BATCH_SIZE = 128 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
EPOCH = 15 
max_len = 512 # model.model.encoder.embed_positions 를 보면 512로 했음을 알 수 있다.
d_model = 512

warmup_steps = 1500 # 데이터 수 * EPOCH / BS = 총 step 수 인것 고려 
LR_scale = 0.5 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 녀석 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

data = pd.read_excel('대화체.xlsx')
custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS, _ = torch.utils.data.random_split(custom_DS, [95000, 2000, 1000, len(custom_DS)-95000-2000-1000])

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_DS))
print(len(val_DS))
print(len(test_DS))

import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

import plotly.graph_objs as go

# Load tokenizer
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')

eos_idx = tokenizer.eos_token_id
pad_idx = tokenizer.pad_token_id
print("eos_idx = ", eos_idx)
print("pad_idx = ", pad_idx)

vocab_size = tokenizer.vocab_size
print(f'tokenizer의 사전 크기: {vocab_size}')

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale

    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate
        
def plot_scheduler(scheduler_name, optimizer, scheduler, total_steps): # LR curve 보기
    lr_history = []
    steps = range(1, total_steps)

    for _ in steps: # base model -> 10만 steps (12시간), big model -> 30만 steps (3.5일) 로 훈련했다고 함
        lr_history += [optimizer.param_groups[0]['lr']]
        scheduler.step()

    plt.figure()
    if total_steps == 100000:
        plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
        plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")    
    plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.35, label="Learning Rate")

    plt.ylim([-0.1*max(lr_history), 1.2*max(lr_history)])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.legend()
    plt.show()
    
optimizer = optim.Adam(nn.Linear(1, 1).parameters(), lr=0)
scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)

        with torch.no_grad():
            # Create a smoothed label distribution
            true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            ignore = target == self.ignore_index
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist.masked_fill_(ignore.unsqueeze(1), 0)

            # Mask to avoid calculating loss for ignored indices
            mask = ~ignore

        # Apply mask and compute loss
        loss = -true_dist * log_probs
        loss = loss.masked_select(mask.unsqueeze(1)).mean()

        return loss

criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1, ignore_index=pad_idx)
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model})은 n_heads ({n_heads})로 나누어 떨어져야 합니다.'

        self.head_dim = d_model // n_heads  # int 형변환 제거

        # 쿼리, 키, 값에 대한 선형 변환
        self.fc_q = nn.Linear(d_model, d_model) 
        self.fc_k = nn.Linear(d_model, d_model) 
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # 어텐션 점수를 위한 스케일 요소
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # 쿼리, 키, 값에 대한 선형 변환 수행
        Q = self.fc_q(Q) 
        K = self.fc_k(K)
        V = self.fc_v(V)

        # 멀티 헤드 어텐션을 위해 텐서 재구성 및 순서 변경
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 스케일드 닷-프로덕트 어텐션 계산
        attention_score = Q @ K.permute(0, 1, 3, 2) / self.scale

        # 마스크 적용 (제공된 경우)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -1e10)

        # 소프트맥스를 사용하여 어텐션 확률 계산
        attention_dist = torch.softmax(attention_score, dim=-1)

        # 어텐션 결과
        attention = attention_dist @ V

        # 어텐션 헤드 재조립
        x = attention.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # 최종 선형 변환
        x = self.fc_o(x)

        return x, attention_dist
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),       ## ADD Dropout
                                    nn.Linear(d_ff, d_model))
    
    def forward(self, x):
        x = self.linear(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        """
        EncoderLayer 클래스의 초기화 메소드입니다.
        :param d_model: 모델의 차원 크기
        :param n_heads: 어텐션 헤드의 개수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        """
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_mask):
        """
        EncoderLayer 클래스의 순전파 메소드입니다.
        :param x: 입력 텐서
        :param enc_mask: 인코더 마스크
        """
        x_norm = self.self_atten_LN(x) ## Pre-LN
        
        # 멀티헤드 어텐션과 잔차 연결
        residual, atten_enc = self.self_atten(x_norm, x_norm, x_norm, enc_mask)
        x = x + self.dropout(residual)

        # 레이어 정규화 적용
        x_norm = self.FF_LN(x)
        # 피드 포워드 네트워크와 잔차 연결
        residual = self.FF(x_norm)
        x = x + self.dropout(residual)

        return x, atten_enc
    
class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        """
        Encoder 클래스의 초기화 메소드입니다.
        :param input_embedding: 입력 임베딩 레이어
        :param max_len: 입력 시퀀스의 최대 길이
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param n_layers: 인코더 레이어의 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        """
        super().__init__()

        # 스케일링 팩터
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        # 인코더 레이어를 n_layers만큼 생성
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])        

    def forward(self, src, mask, atten_map_save=False):
        """
        Encoder 클래스의 순전파 메소드입니다.
        :param src: 입력 소스
        :param mask: 인코더 마스크
        :param atten_map_save: 어텐션 맵 저장 여부
        """
        pos = torch.arange(src.shape[1], device=src.device).repeat(src.shape[0], 1) # 위치 임베딩 생성

        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_encs = []
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save:
                atten_encs.append(atten_enc[0].unsqueeze(0))

        if atten_map_save:
            atten_encs = torch.cat(atten_encs, dim=0)

        return x, atten_encs

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)
        
        self.enc_dec_atten = MHA(d_model, n_heads)
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):

        residual, atten_dec = self.self_atten(x, x, x, dec_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual, atten_enc_dec = self.enc_dec_atten(x, enc_out, enc_out, enc_dec_mask) # Q는 디코더로부터 K,V는 인코더로부터!!
        residual = self.dropout(residual)
        x = self.enc_dec_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_dec, atten_enc_dec
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        """
        DecoderLayer 클래스의 초기화 메소드입니다.
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        """
        super().__init__()

        # Self-Attention 레이어
        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)
        
        # Encoder-Decoder Attention 레이어
        self.enc_dec_atten = MHA(d_model, n_heads)
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)

        # 피드 포워드 네트워크
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        # 드롭아웃
        self.dropout = nn.Dropout(drop_p)

#     def forward(self, x, enc_out, dec_mask, enc_dec_mask):
#         """
#         DecoderLayer 클래스의 순전파 메소드입니다.
#         :param x: 디코더의 입력
#         :param enc_out: 인코더의 출력
#         :param dec_mask: 디코더 마스크
#         :param enc_dec_mask: 인코더-디코더 마스크
#         """
#         # 디코더 Self-Attention
#         residual, atten_dec = self.self_atten(x, x, x, dec_mask)
#         residual = self.dropout(residual)
#         x = self.self_atten_LN(x + residual)

#         # Encoder-Decoder Attention
#         # Q from Decoder + K,V from Encoder
#         residual, atten_enc_dec = self.enc_dec_atten(x, enc_out, enc_out, enc_dec_mask)
#         residual = self.dropout(residual)
#         x = self.enc_dec_atten_LN(x + residual)

#         # 피드 포워드 네트워크
#         residual = self.FF(x)
#         residual = self.dropout(residual)
#         x = self.FF_LN(x + residual)

#         return x, atten_dec, atten_enc_dec

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        """
        DecoderLayer 클래스의 순전파 메소드입니다.
        :param x: 디코더의 입력
        :param enc_out: 인코더의 출력
        :param dec_mask: 디코더 마스크
        :param enc_dec_mask: 인코더-디코더 마스크
        """
        x, atten_dec = self.process_sublayer(x, self.self_atten, self.self_atten_LN, dec_mask)
        x, atten_enc_dec = self.process_sublayer(x, self.enc_dec_atten, self.enc_dec_atten_LN, enc_dec_mask, enc_out)
        x, _ = self.process_sublayer(x, self.FF, self.FF_LN)

        return x, atten_dec, atten_enc_dec

    def process_sublayer(self, x, sublayer, norm_layer, mask=None, enc_out=None):
        """
        디코더의 서브레이어 처리를 위한 함수.
        :param x: 입력 텐서
        :param sublayer: 서브레이어 (어텐션 또는 피드 포워드)
        :param norm_layer: 레이어 정규화
        :param mask: 마스크 (디코더 또는 인코더-디코더 마스크)
        :param enc_out: 인코더의 출력 (인코더-디코더 어텐션에만 필요)
        """
        x_norm = norm_layer(x)
        if isinstance(sublayer, MHA):
            # 멀티헤드 어텐션 레이어의 경우
            if enc_out is not None:
                # 인코더-디코더 어텐션
                residual, atten = sublayer(x_norm, enc_out, enc_out, mask)
            else:
                # 자기 어텐션
                residual, atten = sublayer(x_norm, x_norm, x_norm, mask)
        elif isinstance(sublayer, FeedForward):
            # 피드 포워드 레이어의 경우
            residual = sublayer(x_norm)
            atten = None  # 피드 포워드 레이어는 어텐션 맵을 반환하지 않음
        else:
            raise TypeError("Unsupported sublayer type")

        return x + self.dropout(residual), atten
    
class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        """
        Decoder 클래스의 초기화 메소드.
        :param input_embedding: 입력 임베딩 레이어
        :param max_len: 입력 시퀀스의 최대 길이
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param n_layers: 디코더 레이어의 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        """
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save=False):
        """
        Decoder 클래스의 순전파 메소드.
        :param trg: 타깃 입력
        :param enc_out: 인코더의 출력
        :param dec_mask: 디코더 마스크
        :param enc_dec_mask: 인코더-디코더 마스크
        :param atten_map_save: 어텐션 맵 저장 여부
        """
        pos = torch.arange(trg.shape[1], device=trg.device).repeat(trg.shape[0], 1)

        x = self.scale * self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_decs = []
        atten_enc_decs = []
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save:
                atten_decs.append(atten_dec[0].unsqueeze(0))
                atten_enc_decs.append(atten_enc_dec[0].unsqueeze(0))

        if atten_map_save:
            atten_decs = torch.cat(atten_decs, dim=0)
            atten_enc_decs = torch.cat(atten_enc_decs, dim=0)

        x = self.fc_out(x)
        
        return x, atten_decs, atten_enc_decs

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        """
        Transformer 클래스의 초기화 메소드.
        :param vocab_size: 어휘 사전의 크기
        :param max_len: 입력 시퀀스의 최대 길이
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param n_layers: 인코더 및 디코더 레이어의 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        """
        super().__init__()

        input_embedding = nn.Embedding(vocab_size, d_model) 
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p)
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p)

        self.n_heads = n_heads

        # 파라미터 초기화
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1: 
                nn.init.xavier_uniform_(m.weight) 

    def make_enc_mask(self, src):
        """
        인코더 마스크 생성.
        :param src: 입력 소스 (batch_size, src_len)
        :return: 인코더 마스크 (batch_size, 1, 1, src_len)
                 - pad_idx에 해당하는 위치는 True, 그 외는 False
        """
        enc_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_mask.repeat(1, self.n_heads, src.shape[1], 1).to(src.device)

    def make_dec_mask(self, trg):
        """
        디코더 마스크 생성 (패딩 마스크 및 미래 토큰 마스킹).
        :param trg: 타깃 입력 (batch_size, trg_len)
        :return: 디코더 마스크 (batch_size, 1, trg_len, trg_len)
                 - 패딩 위치 및 미래 위치는 True, 그 외는 False
        """
        trg_pad_mask = (trg == pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(trg.device)
        trg_dec_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1], device=trg.device))==0
        dec_mask = trg_pad_mask | trg_dec_mask
        return dec_mask

    def make_enc_dec_mask(self, src, trg):
        """
        인코더-디코더 마스크 생성.
        :param src: 입력 소스 (batch_size, src_len)
        :param trg: 타깃 입력 (batch_size, trg_len)
        :return: 인코더-디코더 마스크 (batch_size, 1, trg_len, src_len)
                 - 소스의 pad_idx 위치는 True, 그 외는 False
        """
        enc_dec_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(src.device)

    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)

        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)

        return out, atten_encs, atten_decs, atten_enc_decs

save_model_path = './traslator_01.pt'
save_history_path = './translator_history_01.pt'

# Load the tokenizer & input embedding layer & last fc layer
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en')

eos_idx = tokenizer.eos_token_id
pad_idx = tokenizer.pad_token_id
print("eos_idx = ", eos_idx)
print("pad_idx = ", pad_idx)

DEVICE = 'cuda:2'

BATCH_SIZE = 128 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
EPOCH = 150 
max_len = 512 # model.model.encoder.embed_positions 를 보면 512로 했음을 알 수 있다.

scheduler_name = 'Noam'
# scheduler_name = 'Cos'
#### Noam ####
# toatl_steps = 95000 // BATCH_SIZE
warmup_steps = 4000 # 이건 논문에서 제시한 값 (총 10만 step의 4%)
# warmup_steps = 1500 # 데이터 수 * EPOCH / BS = 총 step 수 인것 고려 # 저장된 모델
# warmup_steps = 1300 
LR_scale = 0.5 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 녀석 # 저장된 모델
# LR_scale = 0.4
#### Cos ####
LR_init = 5e-4
T0 = 1500 # 첫 주기
T_mult = 2 # 배 만큼 주기가 길어짐 (1보다 큰 정수여야 함)
#############

vocab_size = tokenizer.vocab_size
print(vocab_size)

# 논문에 나오는 base 모델 (많은 train loss를 많이 줄이려면 많은 Epoch이 요구됨, 또, test 성능도 좋으려면 더 많은 데이터 요구)
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
drop_p = 0.1

# 좀 사이즈 줄인 모델 (훈련된 input_embedding, fc_out 사용하면 사용 불가)
# d_model = 256
# n_heads = 8
# n_layers = 3
# d_ff = 512
# drop_p = 0.1

def Train(model, train_DL, val_DL, criterion, optimizer):
    loss_history = {"train": [], "val": []}
    best_loss = float('inf')

    for ep in range(EPOCH):
        start_time = time.time()  # 에포크 시작 시간 기록

        # 학습 모드
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, optimizer=optimizer, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer)
        loss_history["train"].append(train_loss)

        # 평가 모드
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer)
            loss_history["val"].append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model, "ep": ep, "optimizer": optimizer.state_dict()}, save_model_path)

        # 에포크 소요 시간 계산
        end_time = time.time()
        epoch_time = end_time - start_time

        # 로그 출력
        print(f"Epoch{ep+1}/{EPOCH}|train loss:{train_loss:.5f} val loss:{val_loss:.5f} current_LR:{optimizer.param_groups[0]['lr']:.8f} time:{epoch_time:.2f}s")
        # print("-" * 20)

    # 모든 학습이 끝난 후 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=loss_history["train"], mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=loss_history["val"], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Loss History', xaxis_title='Epoch', yaxis_title='Loss', showlegend=True)
    fig.show()

    torch.save({"loss_history": loss_history, "EPOCH": EPOCH, "BATCH_SIZE": BATCH_SIZE}, save_history_path)

    
def Test(model, test_DL, criterion, max_len, DEVICE, tokenizer):
    model.eval() # test mode로 전환
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer)
    print(f"Test loss: {round(test_loss, 3)} | Test PPL: {round(math.exp(test_loss), 3)}")

def loss_epoch(model, DL, criterion, optimizer=None, max_len=None, DEVICE=None, tokenizer=None):
    N = len(DL.dataset) # 데이터 수

    rloss = 0
    for src_texts, trg_texts in tqdm(DL, leave=False):
        src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = tokenizer(trg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)
        # 추론
        y_hat = model(src, trg[:, :-1])[0] # 모델 통과 시킬 때 trg의 <eos>는 제외!
        # y_hat.shape = 개단채 즉, 훈련 땐 문장이 한번에 튀어나옴
        # 손실
        loss = criterion(y_hat.permute(0, 2, 1), trg[:, 1:]) # 손실 계산 시 <sos> 는 제외!
        # 개단채 -> 개채단으로 바꿔줌 (1D segmentation으로 생각)
        # 업데이트
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # 손실 누적
        loss_b = loss.item() * src.shape[0]
        rloss += loss_b
    loss_e = rloss / N
    return loss_e

model = Transformer(vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p).to(DEVICE)

# params = [p for p in model.parameters() if p.requires_grad] # 사전 학습된 layer를 사용할 경우
params = model.parameters()

# 논문에서 제시한 beta와 eps 사용 & 맨 처음 step 의 LR=0으로 출발 (warm-up)
optimizer = optim.Adam(params, 
                       lr=0, 
                       betas=(0.9, 0.98), 
                       eps=1e-9) 
scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)


Train(model, train_DL, val_DL, criterion, optimizer)