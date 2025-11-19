# Phase 1: AI/ML 기초 강화
## PyTorch와 딥러닝 핵심 개념 마스터 (1개월)

---

## 🎯 목표

이미 2년간 AI를 공부했지만, 음악 AI를 위해 필요한 **실전 PyTorch 능력**과 **딥러닝 핵심 개념**을 확실히 다집니다.

### 완료 기준
- ✅ PyTorch로 모델 설계/학습/저장/로드를 자유자재로
- ✅ CNN, RNN, LSTM, Transformer 구조를 직접 구현 가능
- ✅ Colab에서 GPU를 효율적으로 활용
- ✅ Hugging Face 모델을 파인튜닝할 수 있음

---

## 📅 주차별 학습 계획

### Week 1: PyTorch 핵심 마스터
**목표**: Tensor 연산부터 모델 학습까지 완전 이해

#### Day 1-2: PyTorch Basics
- Tensor 생성 및 연산
- Autograd와 역전파 이해
- GPU 사용법 (CUDA)

**실습**: `01_pytorch_basics.py`
```python
# Tensor 연산 연습
# Autograd 동작 원리 실험
# GPU vs CPU 속도 비교
```

#### Day 3-4: Neural Network 구조
- nn.Module 상속
- Forward/Backward pass
- Loss function과 Optimizer

**실습**: `02_neural_network.py`
```python
# 간단한 MLP 구현
# MNIST 분류기
# Custom loss function
```

#### Day 5-7: Training Loop & Best Practices
- DataLoader와 Dataset
- Training/Validation split
- 모델 저장 및 로드
- TensorBoard 사용

**실습**: `03_training_loop.py`
```python
# 완전한 학습 파이프라인
# Early stopping
# Learning rate scheduler
# Checkpoint 관리
```

---

### Week 2: CNN & Computer Vision
**목표**: 이미지 처리 기술을 음악 스펙트로그램에 응용

#### Day 8-10: CNN 구조
- Convolution 연산 원리
- Pooling, Padding, Stride
- ResNet, EfficientNet 구조 분석

**실습**: `04_cnn_basics.py`
```python
# Custom CNN 구현
# CIFAR-10 분류
# Feature map 시각화
```

#### Day 11-14: Transfer Learning
- 사전학습 모델 활용
- Fine-tuning 전략
- Feature extraction vs Full fine-tuning

**실습**: `05_transfer_learning.py`
```python
# ResNet50 파인튜닝
# Custom dataset 학습
# Gradual unfreezing
```

**음악 AI 연결**:
- Mel-spectrogram은 이미지처럼 처리
- CNN으로 음악 장르 분류 가능
- 이미지 생성 기법 → 스펙트로그램 생성

---

### Week 3: RNN, LSTM, Sequence Modeling
**목표**: 시퀀스 데이터 처리 (음악의 핵심!)

#### Day 15-17: RNN/LSTM 이론과 구현
- RNN의 한계와 LSTM의 해결책
- GRU vs LSTM
- Bidirectional RNN

**실습**: `06_rnn_lstm.py`
```python
# Vanilla RNN 구현
# LSTM for sequence prediction
# 텍스트 생성 (Character-level)
```

#### Day 18-21: Sequence-to-Sequence
- Encoder-Decoder 구조
- Attention mechanism
- Teacher forcing

**실습**: `07_seq2seq.py`
```python
# 간단한 번역 모델
# Attention 시각화
# Beam search 구현
```

**음악 AI 연결**:
- MIDI는 시퀀스 데이터
- 멜로디 생성 = 시퀀스 생성
- 코드 진행도 시퀀스

---

### Week 4: Transformer & Modern Architectures
**목표**: 음악 생성의 최신 기술

#### Day 22-25: Transformer 이해
- Self-attention 메커니즘
- Multi-head attention
- Positional encoding
- Layer normalization

**실습**: `08_transformer.py`
```python
# Transformer from scratch
# Self-attention 시각화
# 간단한 언어 모델
```

#### Day 26-28: VAE & Generative Models
- VAE 원리 (Reparameterization trick)
- Latent space 조작
- Conditional VAE

**실습**: `09_vae.py`
```python
# MNIST VAE
# Latent space interpolation
# Conditional generation
```

**음악 AI 연결**:
- Music Transformer는 이 구조 기반
- MusicVAE는 음악 latent space 학습
- 스타일 전이에 핵심 기술

---

## 📚 학습 자료

### 필수 강의
1. **PyTorch Tutorials** (공식):
   - https://pytorch.org/tutorials/
   - Beginner부터 Advanced까지

2. **Fast.ai - Practical Deep Learning**:
   - https://course.fast.ai/
   - 실전 중심, Top-down 접근

3. **Stanford CS231n** (CNN):
   - http://cs231n.stanford.edu/
   - Computer Vision 기초

4. **Stanford CS224n** (NLP/Transformer):
   - http://web.stanford.edu/class/cs224n/
   - Sequence modeling 심화

### 추천 도서
- **"Deep Learning with PyTorch"** (Stevens et al.)
- **"Dive into Deep Learning"** (d2l.ai) - 무료 온라인

### 논문 (선택)
- "Attention Is All You Need" (Transformer)
- "Auto-Encoding Variational Bayes" (VAE)

---

## 💻 실습 프로젝트

### Project 1: MNIST 분류기 (CNN)
**난이도**: ⭐⭐☆☆☆

```python
# 목표: 99% 정확도 달성
- Custom CNN 설계
- Data augmentation
- TensorBoard 시각화
```

**코드**: `projects/01_mnist_classifier.py`

---

### Project 2: 텍스트 생성기 (LSTM)
**난이도**: ⭐⭐⭐☆☆

```python
# 목표: 셰익스피어 스타일 텍스트 생성
- Character-level LSTM
- Temperature sampling
- Top-k sampling
```

**코드**: `projects/02_text_generator.py`

**음악 연결**: MIDI note도 character처럼 처리 가능!

---

### Project 3: 이미지 생성기 (VAE)
**난이도**: ⭐⭐⭐☆☆

```python
# 목표: MNIST 손글씨 생성
- VAE 구현
- Latent space 탐험
- Conditional generation (특정 숫자 생성)
```

**코드**: `projects/03_vae_generator.py`

**음악 연결**: MusicVAE의 기초!

---

### Project 4: Sentiment Analysis (Transformer)
**난이도**: ⭐⭐⭐⭐☆

```python
# 목표: IMDB 리뷰 감성 분석
- Mini Transformer 구현
- Hugging Face 모델 파인튜닝
- Attention weight 시각화
```

**코드**: `projects/04_sentiment_transformer.py`

---

## 🛠️ 환경 설정

### 로컬 환경 (선택)
```bash
# Python 3.10 이상
python --version

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### Google Colab (추천)
```python
# 새 노트북 생성
# 런타임 → 런타임 유형 변경 → GPU (T4)

# 필수 라이브러리 설치
!pip install torch torchvision torchaudio
!pip install tensorboard
!pip install matplotlib seaborn
```

---

## 📊 학습 진도 체크리스트

### Week 1: PyTorch Basics ✅
- [ ] Tensor 연산 100% 이해
- [ ] Autograd 원리 설명 가능
- [ ] nn.Module로 모델 설계 가능
- [ ] MNIST 95% 정확도 달성

### Week 2: CNN ✅
- [ ] Convolution 연산 손으로 계산 가능
- [ ] ResNet 구조 설명 가능
- [ ] Transfer learning 성공 경험
- [ ] CIFAR-10 80% 정확도

### Week 3: RNN/LSTM ✅
- [ ] LSTM cell 구조 그릴 수 있음
- [ ] Sequence prediction 구현
- [ ] Attention mechanism 이해
- [ ] 텍스트 생성기 완성

### Week 4: Transformer & VAE ✅
- [ ] Self-attention 수식 이해
- [ ] Transformer 처음부터 구현
- [ ] VAE latent space 조작
- [ ] 4개 프로젝트 모두 완료

---

## 🎯 평가 기준

### 이론 이해도 (40%)
- [ ] PyTorch 핵심 개념 설명 가능
- [ ] CNN, RNN, Transformer 차이점 명확히 알기
- [ ] Loss, Optimizer, Regularization 이해

### 코드 구현 능력 (40%)
- [ ] 처음부터 모델 설계 가능
- [ ] 디버깅 능력
- [ ] 코드 가독성 및 문서화

### 실전 적용 (20%)
- [ ] Colab에서 GPU 효율적 사용
- [ ] Hugging Face 모델 다루기
- [ ] 실험 결과 시각화 및 분석

---

## 💡 학습 팁

### DO ✅
1. **손으로 코딩**: 복사-붙여넣기 금지, 직접 타이핑
2. **작게 시작**: 간단한 예제부터 → 복잡한 프로젝트
3. **시각화**: TensorBoard, matplotlib으로 이해 확인
4. **문서 읽기**: PyTorch 공식 문서 습관화
5. **디버깅 연습**: pdb, print문 적극 활용

### DON'T ❌
1. **이론만**: 코드 없이 논문만 읽지 말기
2. **완벽주의**: 100% 이해 후 넘어가려 하지 말기
3. **고립**: 막히면 Stack Overflow, Discord 활용
4. **GPU 낭비**: 디버깅은 CPU로, 학습만 GPU로

---

## 🔗 다음 단계 연결

Phase 1을 완료하면:
- ✅ **PyTorch 능숙도**: 음악 모델 구현 준비 완료
- ✅ **Sequence modeling**: MIDI 생성 기술 획득
- ✅ **VAE**: MusicVAE 이해를 위한 기반
- ✅ **Transformer**: Music Transformer 학습 준비

**➡️ [Phase 2: Audio/MIDI Processing](../phase2-audio-processing/learning-guide.md)**

이제 음악 데이터를 다룰 준비가 되었습니다!

---

## 📞 도움이 필요할 때

### 커뮤니티
- **PyTorch Forums**: discuss.pytorch.org
- **r/MachineLearning**: Reddit
- **Discord**: PyTorch KR

### 질문 전 체크리스트
1. 에러 메시지 전체 복사
2. 최소 재현 코드 작성
3. 시도한 해결책 정리
4. 환경 정보 (PyTorch 버전, GPU 등)

---

**"Phase 1은 음악 AI의 기초 체력 다지기입니다. 탄탄한 기본기는 이후 모든 단계를 수월하게 만듭니다."**

*Estimated Time: 30일 (하루 2-3시간)*
*Difficulty: ⭐⭐⭐☆☆*
*Next: Phase 2 - Audio Processing* 🎵
