import pandas as pd
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import MarianTokenizer
import pandas as pd
from tqdm import tqdm
import plotly.graph_objs as go

import click


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale=1):
        self.optimizer = optimizer  # 최적화할 옵티마이저
        self.step_count = 0  # 현재까지 진행된 스텝 수
        self.d_model = d_model  # 모델의 차원 수
        self.warmup_steps = warmup_steps  # 웜업 단계에서의 스텝 수
        self.LR_scale = LR_scale  # 학습률 스케일 인자
        self._d_model_factor = self.LR_scale * (self.d_model ** -0.5)  # 모델 차원에 대한 계수를 미리 계산

    def step(self):
        self.step_count += 1  # 스텝 수 증가
        lr = self.calculate_learning_rate()  # 새 학습률 계산
        self.optimizer.param_groups[0]['lr'] = lr  # 옵티마이저의 학습률 갱신

    def calculate_learning_rate(self):
        # 초기 웜업 단계에서는 학습률을 서서히 증가시키고, 이후에는 감소시키는 방식으로 계산
        minimum_factor = min(self.step_count ** -0.5, self.step_count * self.warmup_steps ** -1.5)
        return self._d_model_factor * minimum_factor
        

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
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_mask):
        """
        EncoderLayer 클래스의 순전파 메소드입니다.
        :param x: 입력 텐서
        :param enc_mask: 인코더 마스크
        """
        x_norm = self.LN(x) ## Pre-LN
        
        # 멀티헤드 어텐션과 잔차 연결
        output, atten_enc = self.self_atten(x_norm, x_norm, x_norm, enc_mask)
        x = x + self.dropout(output)

        # 레이어 정규화 적용
        x_norm = self.LN(x)
        # 피드 포워드 네트워크와 잔차 연결
        output = self.FF(x_norm)
        x = x_norm + self.dropout(output)
        # x = self.LN(x)

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
        self.LN = nn.LayerNorm(d_model) ## encoder 레이어 후에 LN 적용

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

        x = self.LN(x)
        return x, atten_encs

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
        self.atten = MHA(d_model, n_heads) # Attention for Self & Cross
        self.FF = FeedForward(d_model, d_ff, drop_p) # ff network
        self.LN = nn.LayerNorm(d_model) # Layer Normalization
        self.dropout = nn.Dropout(drop_p) # Dropout

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        """
        DecoderLayer 클래스의 순전파 메소드입니다.
        :param x: 디코더의 입력
        :param enc_out: 인코더의 출력
        :param dec_mask: 디코더 마스크
        :param enc_dec_mask: 인코더-디코더 마스크
        """
        x, atten_dec = self.process_sublayer(x, self.atten, self.LN, dec_mask)
        x, atten_enc_dec = self.process_sublayer(x, self.atten, self.LN, enc_dec_mask, enc_out)
        x, _ = self.process_sublayer(x, self.FF, self.LN)

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
        if isinstance(sublayer, MHA): # mha case
            if enc_out is not None: # encoder-decoder attention
                residual, atten = sublayer(x_norm, enc_out, enc_out, mask)
            else: # self attention
                residual, atten = sublayer(x_norm, x_norm, x_norm, mask)
        elif isinstance(sublayer, FeedForward): # ff network
            residual = sublayer(x_norm)
            atten = None  # 피드 포워드 레이어는 어텐션 맵을 반환하지 않음
        else:
            raise TypeError("Unsupported sublayer type")

        return x + self.dropout(residual), atten
    
class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size):
        """
        Decoder 클래스의 초기화 메소드.
        :param input_embedding: 입력 임베딩 레이어
        :param max_len: 입력 시퀀스의 최대 길이
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param n_layers: 디코더 레이어의 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        :param vocab_size: 사전의 크기
        """
        super().__init__()        
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])
        self.LN = nn.LayerNorm(d_model)
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

        x = self.LN(x) ## decoder layers 이후 LN
        x = self.fc_out(x)
        
        return x, atten_decs, atten_enc_decs

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx):
        """
        Transformer 클래스의 초기화 메소드.
        :param vocab_size: 어휘 사전의 크기
        :param max_len: 입력 시퀀스의 최대 길이
        :param d_model: 모델의 차원 크기
        :param n_heads: 멀티헤드 어텐션의 헤드 수
        :param n_layers: 인코더 및 디코더 레이어의 수
        :param d_ff: 피드 포워드 네트워크의 내부 차원
        :param drop_p: 드롭아웃 비율
        :param pad_idx: padding token의 index
        """
        super().__init__()
        self.pad_idx = pad_idx
        input_embedding = nn.Embedding(vocab_size, d_model) 
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p)
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size)

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
        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_mask.repeat(1, self.n_heads, src.shape[1], 1).to(src.device)

    def make_dec_mask(self, trg):
        """
        디코더 마스크 생성 (패딩 마스크 및 미래 토큰 마스킹).
        :param trg: 타깃 입력 (batch_size, trg_len)
        :return: 디코더 마스크 (batch_size, 1, trg_len, trg_len)
                 - 패딩 위치 및 미래 위치는 True, 그 외는 False
        """
        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2)
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
        enc_dec_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(src.device)

    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)

        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)

        return out, atten_encs, atten_decs, atten_enc_decs

def Train(model, 
          train_DL, 
          val_DL, 
          criterion, 
          optimizer,
          params):
    
    BATCH_SIZE = params['batch_size']
    EPOCH = params['epoch']
    max_len = params['max_len']
    DEVICE = params['device']
    tokenizer = params['tokenizer']
    save_model_path = params['save_model_path']
    save_history_path = params['save_history_path']
    scheduler = params['scheduler']
    
    history = {"train": [], "val": [], "lr":[]}
    best_loss = float('inf')

    for ep in range(EPOCH):
        start_time = time.time()  # 에포크 시작 시간 기록

        # 학습 모드
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, optimizer=optimizer, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer, scheduler=scheduler)
        history["train"].append(train_loss)

        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        # 평가 모드
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer)
            history["val"].append(val_loss)
            epoch_time = time.time() - start_time

            # 로그 출력
            if val_loss < best_loss:
                best_loss = val_loss
                # loss_path = save_model_path + f'_loss{val_loss:.4f}.pt'
                loss_path = save_model_path + '.pt'
                torch.save({"model": model, "ep": ep, "optimizer": optimizer.state_dict(), 'loss':val_loss}, loss_path)
                print(f"| Epoch {ep+1}/{EPOCH} | train loss:{train_loss:.5f} val loss:{val_loss:.5f} current_LR:{optimizer.param_groups[0]['lr']:.8f} time:{epoch_time:.2f}s => Model Saved!")
            else :
                print(f"| Epoch {ep+1}/{EPOCH} | train loss:{train_loss:.5f} val loss:{val_loss:.5f} current_LR:{optimizer.param_groups[0]['lr']:.8f} time:{epoch_time:.2f}s")

    torch.save({"loss_history": history, "EPOCH": EPOCH, "BATCH_SIZE": BATCH_SIZE}, save_history_path)
    
    show_history(history=history, EPOCH=EPOCH, save_path=save_model_path)
    
def show_history(history, EPOCH, save_path='train_history_ls'):
    # train loss, val loss 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history["train"], mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history["val"], mode='lines+markers', name='Validation Loss'))

    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis=dict(title='Loss'),
        showlegend=True
    )
    fig.write_image(save_path+"loss.png")
    # fig.show()
    
    # learning rate 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history['lr'], mode='lines+markers', name='Learning Rate'))

    # 레이아웃 업데이트
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis=dict(title='Learning Rate'),
        showlegend=True
    )
    fig.write_image(save_path+"lr.png")
    # fig.show()
    

def loss_epoch(model, DL, criterion, optimizer=None, max_len=None, DEVICE=None, tokenizer=None, scheduler=None):
    N = len(DL.dataset) # 데이터 수

    rloss = 0
    for src_texts, trg_texts in tqdm(DL, leave=False):
        src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = tokenizer(trg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)

        # inference
        y_hat = model(src, trg[:, :-1])[0] # 모델 통과 시킬 때 trg의 <eos>는 제외!
        loss = criterion(y_hat.permute(0, 2, 1), trg[:, 1:]) # 손실 계산 시 <sos> 는 제외!
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
                    
        # loss accumulation
        loss_b = loss.item() * src.shape[0]
        rloss += loss_b

    loss_e = rloss / N
    return loss_e

@click.command()
@click.option('--batch', default=128, help='batch size')
@click.option('--epoch', default=100, help='train epoch')
@click.option('--device', default='cuda:0', help='cuda:index')
@click.option('--model_size', default='small', help='select among [base] or [small]')
@click.option('--criterion_type', default='ce', help='select among [ce] or [lsce]')
@click.option('--label_smoothing', default=0.1, help='ratio of label smoothing')
def main(batch:int=128, 
               epoch:int=100, 
               device:str='cuda:0', 
               model_size:str='small',
               criterion_type:str='ce',
               label_smoothing:float=0.1,
               ):
    
    text = "Train Transformer Translator Kor-En is Started!"
    styled_text = click.style(text, fg='green', bold=True)
    click.echo(styled_text)    
    
    params = dict()
    params['batch_size'] = BATCH_SIZE = batch ## 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
    params['epoch'] = epoch ## 논문에선 약 560 에포크(10만 스탭) 진행

    params['save_model_path'] = f'./results/translator_{criterion_type}' if criterion_type=='ce' else f'./results/translator_{criterion_type}{label_smoothing}'
    params['save_history_path'] = f'./results/translator_history_{criterion_type}.pt' if criterion_type=='ce' else f'./results/translator_history_{criterion_type}{label_smoothing}.pt'

    params['device'] = DEVICE = device 
    
    if model_size == 'base':
        # 논문에 나오는 base 모델
        params['max_len'] = max_len = 512
        d_model = 512
        n_heads = 8
        n_layers = 6
        d_ff = 2048
        drop_p = 0.1
        warmup_steps = 4000 
        LR_scale = 1 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 스케일 -> 최초 논문엔 언급X, 후속논문에 등장
    elif model_size == 'small':
        # Small 모델
        params['max_len'] = max_len = 80
        d_model = 256
        n_heads = 8
        n_layers = 3
        d_ff = 512
        drop_p = 0.1
        # warmup_steps = int((99000 / BATCH_SIZE) * 0.05 * epoch) ## 논문에선 4,000 스탭 
        warmup_steps = 1500
        LR_scale = 2 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 스케일 -> 최초 논문엔 언급X, 후속논문에 등장
    else :
        raise "model size should be selected in ['base', 'small']"
    
    # Load tokenizer
    params['tokenizer'] = tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    pad_idx = tokenizer.pad_token_id
    
    if criterion_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    elif criterion_type == 'lsce':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=pad_idx)
    
    text = 'All params are defined!'
    styled_text = click.style(text, fg='cyan', bold=True)
    click.echo(styled_text)
    click.echo(params)
    
    ### Train Start ###

    data = pd.read_excel('대화체.xlsx')
    custom_DS = CustomDataset(data)

    train_DS, val_DS= torch.utils.data.random_split(custom_DS, [99000, 1000])

    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)

    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    model = Transformer(vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx).to(DEVICE)

    # 논문에서 제시한 beta와 eps 사용 & 맨 처음 step 의 LR=0으로 출발 (warm-up)
    optimizer = optim.Adam(model.parameters(), 
                        lr=0, 
                        betas=(0.9, 0.98), 
                        eps=1e-9,
                        weight_decay=1e-5, ## l2-Regularization
                        ) 
    params['scheduler'] = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)

    Train(model, train_DL, val_DL, criterion, optimizer, params)

    text = 'Train is done!'
    styled_text = click.style(text, fg='cyan', bold=True)
    click.echo(styled_text)

if __name__ == "__main__":    

    main()