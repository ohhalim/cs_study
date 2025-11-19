# Phase 4: 음악 생성 모델 학습
## 최신 Music Generation Models 마스터 (3개월)

---

## 🎯 목표

최신 음악 생성 모델들을 이해하고 직접 학습하여, Charlie Parker AI의 기술적 기반을 마련합니다.

### 완료 기준
- ✅ Music Transformer 구현 및 학습
- ✅ MusicVAE로 latent space 이해
- ✅ MusicGen 파인튜닝 성공
- ✅ 각 모델의 장단점 파악
- ✅ Charlie Parker 데이터로 실험 완료

---

## 📅 모델별 학습 계획

### Month 1: Music Transformer

#### 1주차: 논문 리뷰 & 구조 이해
**논문**: "Music Transformer" (Huang et al., 2018)

**핵심 개념**:
- Relative positional encoding (음악의 상대적 위치 중요)
- Autoregressive generation
- Event-based representation

**실습**:
```python
# code/01_music_transformer.py
- Transformer encoder 구현
- MIDI → Event representation
- Relative attention 구현
```

#### 2-4주차: 구현 & 학습
```python
# 구현 단계:
1. MIDI tokenization (pitch, velocity, time)
2. Transformer architecture
3. Training loop
4. Generation (temperature, top-k sampling)

# 학습 데이터:
- 100개 Charlie Parker MIDI
- Augmentation으로 1000개 확장

# 하이퍼파라미터:
- d_model: 512
- num_heads: 8
- num_layers: 6
- sequence_length: 2048
- batch_size: 8 (Gradient accumulation)
```

**목표 결과**: 32-bar 재즈 솔로 생성

---

### Month 2: MusicVAE

#### 1주차: VAE 이론
**논문**: "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music" (Roberts et al., 2018)

**핵심 개념**:
- Variational Autoencoder for music
- Hierarchical decoder
- Latent space interpolation
- Conductor model

#### 2-4주차: 구현 & 실험
```python
# code/02_music_vae.py
- Encoder: MIDI → Latent vector (Z)
- Decoder: Z → MIDI
- KL divergence loss + Reconstruction loss

# 실험:
1. Charlie Parker 솔로 → latent space
2. Interpolation (파커 스타일 A ↔ B)
3. Style transfer (Parker → Coltrane)
4. Latent space arithmetic (Parker + Blues = ?)
```

**Magic**:
- 2개 Parker 솔로 사이 interpolation
- 새로운, 하지만 Parker 스타일인 솔로 생성!

---

### Month 3: MusicGen (Meta)

#### 1-2주차: MusicGen 이해
**논문**: "Simple and Controllable Music Generation" (Copet et al., 2023)

**핵심**:
- EnCodec: Audio compression
- Transformer LM
- Text conditioning (optional)
- Audio generation (not MIDI!)

#### 3-4주차: Fine-tuning
```python
# Hugging Face 사용
from transformers import MusicgenForConditionalGeneration

# 1. Pre-trained model 로드
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# 2. Charlie Parker 오디오 데이터 준비
# 3. Fine-tuning
# 4. "Charlie Parker jazz solo" → Audio 생성!
```

**과제**:
- Charlie Parker 녹음 50개로 파인튜닝
- "Be-bop jazz solo in F" 프롬프트로 생성
- 품질 평가

---

## 💻 핵심 코드 예시

### Music Transformer 생성
```python
import torch
from music_transformer import MusicTransformer

# 모델 로드
model = MusicTransformer(
    vocab_size=128,  # MIDI notes
    d_model=512,
    num_heads=8,
    num_layers=6
).cuda()

# 시작 토큰 (F major chord)
start = torch.tensor([[60, 64, 67]]).cuda()  # F A C

# 생성
generated = model.generate(
    start_tokens=start,
    max_len=512,
    temperature=0.9,
    top_k=40
)

# MIDI 저장
save_to_midi(generated, "parker_ai_v1.mid")
```

### MusicVAE Interpolation
```python
from music_vae import MusicVAE

model = MusicVAE(latent_dim=512)

# 2개 Parker 솔로
solo_A = load_midi("ornithology_solo.mid")
solo_B = load_midi("confirmation_solo.mid")

# Encode
z_A = model.encode(solo_A)
z_B = model.encode(solo_B)

# Interpolate
results = []
for alpha in np.linspace(0, 1, 9):
    z_interp = (1 - alpha) * z_A + alpha * z_B
    solo_interp = model.decode(z_interp)
    results.append(solo_interp)

# 9개 새로운 솔로!
```

---

## 📊 모델 비교

| 모델 | 장점 | 단점 | Charlie Parker 적합도 |
|------|------|------|---------------------|
| **Music Transformer** | - 긴 시퀀스 학습<br>- 정교한 패턴 | - 학습 느림<br>- GPU 많이 사용 | ⭐⭐⭐⭐⭐ 최적 |
| **MusicVAE** | - Latent space 조작<br>- Interpolation | - 짧은 시퀀스 (2-4 bar)<br>- 구조 제한적 | ⭐⭐⭐☆☆ 실험용 |
| **MusicGen** | - Audio 직접 생성<br>- Text conditioning | - 파인튜닝 어려움<br>- MIDI 아님 | ⭐⭐⭐⭐☆ 데모용 |
| **Jukebox** | - 고품질 오디오<br>- 장시간 생성 | - 매우 느림<br>- 리소스 많이 필요 | ⭐⭐☆☆☆ 참고용 |

**추천**: Music Transformer를 메인으로, MusicVAE는 실험용

---

## 🎯 학습 전략

### GPU 리소스 관리
```python
# Mixed Precision Training (메모리 50% 절약)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient Accumulation (배치 크기 늘림)
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 학습 팁
1. **작게 시작**: 소규모 모델로 검증 → 확장
2. **체크포인트**: 매 epoch 저장 (Colab 타임아웃 대비)
3. **TensorBoard**: Loss, 샘플 생성 모니터링
4. **Early stopping**: Validation loss로 과적합 방지

---

## 📈 평가 지표

### 정량적 평가
1. **Perplexity**: 낮을수록 좋음
2. **Note Accuracy**: Ground truth와 비교
3. **Pitch Entropy**: 음 다양성
4. **Rhythm Diversity**: 리듬 복잡도

### 정성적 평가
1. **Blind Test**: 재즈 뮤지션에게
   - "이것이 Charlie Parker인가?"
   - 50% 이상이면 성공!

2. **Musical Coherence**:
   - 프레이즈 길이 적절한가?
   - 코드 진행 따르는가?
   - Be-bop 특징 있는가?

---

## 🔗 다음 단계 연결

Phase 4를 완료하면:
- ✅ **Music Transformer**: Charlie Parker 스타일 생성 가능
- ✅ **MusicVAE**: 스타일 조작 경험
- ✅ **MusicGen**: 오디오 생성 가능
- ✅ **기술 스택**: 실전 프로젝트 준비 완료

**➡️ [Phase 5: Charlie Parker AI Project](../phase5-charlie-parker-ai/learning-guide.md)**

이제 본격적으로 BirdAI를 만들 차례!

---

**"모델은 도구입니다. Charlie Parker의 정신을 담는 것이 목표입니다."**

*Estimated Time: 90일 (하루 3-4시간)*
*Difficulty: ⭐⭐⭐⭐⭐*
*Next: Phase 5 - BirdAI Project* 🐦
