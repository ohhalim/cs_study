# 음악 AI 리소스 가이드
## 데이터셋, GPU, 도구, 커뮤니티 총정리

---

## 📦 데이터셋

### MIDI 데이터셋

#### 1. Lakh MIDI Dataset
- **규모**: 176,581개 MIDI 파일
- **장르**: 다양 (팝, 클래식, 재즈 등)
- **다운로드**: https://colinraffel.com/projects/lmd/
- **용도**: Pre-training, 일반 음악 학습

#### 2. MAESTRO Dataset
- **규모**: 200시간 클래식 피아노
- **특징**: 고품질, 정렬된 오디오+MIDI
- **다운로드**: https://magenta.tensorflow.org/datasets/maestro
- **용도**: 피아노 생성 모델

#### 3. Jazz MIDI Collection
- **소스**:
  - r/jazzmidi (Reddit)
  - https://www.mfiles.co.uk/jazz-midi-files.htm
  - https://freejazzlessons.com
- **특징**: 재즈 스탠다드, 트랜스크립션
- **용도**: 재즈 학습 (Charlie Parker 포함!)

#### 4. Charlie Parker 전용
**직접 수집 필요**:
1. **YouTube**: "Charlie Parker solo transcription"
2. **The Omnibook**: PDF 악보 → MuseScore → MIDI
3. **Jazz Transcriptions**: https://jazzstudiesonline.org

**추천 곡** (데이터 수집 우선순위):
1. Ornithology
2. Confirmation
3. Ko-Ko
4. Billie's Bounce
5. Now's the Time
6. Anthropology
7. Scrapple from the Apple
8. Yardbird Suite
9. Donna Lee
10. Au Privave

---

### 오디오 데이터셋

#### 1. MagnaTagATune
- **규모**: 25,863곡
- **특징**: 태그 레이블링
- **다운로드**: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- **용도**: 음악 분류, 특징 추출

#### 2. Free Music Archive (FMA)
- **규모**: 106,574곡
- **특징**: 다양한 장르, 메타데이터
- **다운로드**: https://github.com/mdeff/fma
- **용도**: 장르 분류, 일반 음악 이해

#### 3. NSynth Dataset
- **규모**: 305,979개 음표 (4초)
- **특징**: 악기별 단일 음표
- **다운로드**: https://magenta.tensorflow.org/datasets/nsynth
- **용도**: 음색 생성, 신디사이저

---

## 💻 GPU 리소스 관리

### 무료 옵션

#### 1. Google Colab
- **GPU**: T4 (16GB)
- **무료 한도**: 주 15-20시간
- **장점**: 설정 필요 없음, Jupyter 환경
- **단점**: 세션 타임아웃 (12시간), 불안정
- **팁**:
  ```python
  # 백그라운드 탭 유지 (콘솔에서 실행)
  function ClickConnect(){
    console.log("연결 유지");
    document.querySelector("colab-connect-button").click()
  }
  setInterval(ClickConnect, 60000)
  ```

#### 2. Kaggle Notebooks
- **GPU**: P100 (16GB), T4 (16GB)
- **무료 한도**: 주 30시간
- **장점**: Colab보다 안정적
- **단점**: 인터넷 사용 제한
- **추천**: 데이터셋 학습용

#### 3. Lightning AI (구 Grid.ai)
- **GPU**: T4, A10
- **무료 한도**: 월 22시간
- **장점**: 프로덕션급 환경
- **단점**: 복잡한 설정

#### 4. Paperspace Gradient
- **GPU**: M4000
- **무료 한도**: 제한적
- **장점**: Jupyter 환경
- **단점**: 느림

---

### 유료 옵션 (가성비 순)

#### 1. Google Colab Pro ($10/월)
- **GPU**: V100, A100
- **한도**: 100 compute units
- **장점**:
  - 백그라운드 실행
  - 더 긴 세션 (24시간)
  - 우선 순위 GPU
- **추천 대상**: 개인 프로젝트, 학생

#### 2. RunPod (~$0.2-0.5/시간)
- **GPU**: RTX 3090, RTX 4090, A6000
- **장점**:
  - 사용한 만큼만 지불
  - 다양한 GPU 선택
  - SSH 접속 가능
- **팁**:
  ```bash
  # Spot instance (50% 저렴)
  # Community cloud (더 저렴)
  ```
- **추천 대상**: Phase 4-5 집중 학습

#### 3. Lambda Labs ($0.5-1.5/시간)
- **GPU**: A100 (40GB/80GB)
- **장점**: 안정적, 빠른 네트워크
- **단점**: 비쌈
- **추천 대상**: 대규모 학습, 마지막 단계

#### 4. Vast.ai (~$0.1-0.3/시간)
- **GPU**: 개인이 임대하는 GPU
- **장점**: 매우 저렴
- **단점**: 불안정, 복잡
- **추천 대상**: 숙련자, 실험용

---

### 비용 절감 전략

#### 코드 최적화
```python
# 1. Mixed Precision Training (메모리 50% 절약)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. Gradient Accumulation (작은 배치로 큰 효과)
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Gradient Checkpointing (메모리 70% 절약, 속도 20% 감소)
from torch.utils.checkpoint import checkpoint

def forward(x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

#### 학습 전략
1. **작은 모델로 실험**: CPU에서 디버깅 → GPU로 학습
2. **체크포인트 자주 저장**: 세션 타임아웃 대비
3. **오프피크 시간 활용**: RunPod 가격 변동
4. **Spot instance**: 50% 저렴 (중단 위험 있음)

#### 예상 비용 (12개월 프로젝트)

| 단계 | GPU 시간 | 서비스 | 예상 비용 |
|------|---------|--------|----------|
| Phase 1-3 | 20시간 | Colab (무료) | $0 |
| Phase 4 실험 | 40시간 | Colab Pro | $10 |
| Phase 4 본격 | 80시간 | RunPod (RTX 3090) | $20 |
| Phase 5 학습 | 120시간 | RunPod (RTX 4090) | $60 |
| **총계** | 260시간 | - | **$90** |

**💡 팁**: 월 $10 이하로도 충분히 가능! (무료 GPU 활용 시)

---

## 🛠️ 필수 도구 & 라이브러리

### 딥러닝 프레임워크
```bash
# PyTorch (필수)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers (Hugging Face)
pip install transformers datasets accelerate

# Lightning (고급 학습)
pip install pytorch-lightning
```

### MIDI 처리
```bash
pip install pretty-midi        # 가장 직관적
pip install mido               # Low-level 제어
pip install music21            # 음악 이론 분석
pip install pyfluidsynth       # MIDI → Audio 변환
```

### 오디오 처리
```bash
pip install librosa            # 필수! 오디오 분석
pip install soundfile          # I/O
pip install basic-pitch        # Audio → MIDI (Spotify)
pip install torchaudio         # PyTorch 통합
pip install audioread          # 다양한 포맷 지원
```

### 시각화
```bash
pip install matplotlib seaborn
pip install librosa[display]   # Spectrogram 시각화
pip install tensorboard        # 학습 모니터링
pip install wandb              # 실험 트래킹 (선택)
```

### 웹 데모
```bash
pip install gradio             # ML 데모 (추천!)
pip install streamlit          # 데이터 앱
pip install fastapi uvicorn    # API 서버
```

### 유틸리티
```bash
pip install tqdm               # Progress bar
pip install python-dotenv      # 환경변수
pip install pyyaml             # Config 파일
```

---

## 📚 학습 자료

### 온라인 강의

#### 딥러닝 기초
1. **Fast.ai - Practical Deep Learning**
   - 링크: https://course.fast.ai/
   - 난이도: 초급-중급
   - 무료, 실전 중심

2. **Stanford CS231n - CNN**
   - 링크: http://cs231n.stanford.edu/
   - 난이도: 중급
   - 무료, 비디오 + 과제

3. **Stanford CS224n - NLP**
   - 링크: http://web.stanford.edu/class/cs224n/
   - 난이도: 중급
   - Transformer 심화

#### 음악 AI 전문
1. **Coursera - Audio Signal Processing for ML**
   - 강사: Xavier Serra (UPF Barcelona)
   - 난이도: 중급
   - 유료 (재정 지원 가능)

2. **Magenta Tutorials**
   - 링크: https://magenta.tensorflow.org/
   - 난이도: 초급-중급
   - 무료, 코드 중심

3. **MIT 6.S192 - Deep Learning for Art**
   - 링크: https://www.youtube.com/playlist?list=PLCpMvp7ftsnIbNwRnQJbDNRqO6qiN3EyH
   - 난이도: 중급
   - 음악 포함

### 논문 (필수)

#### Music Transformer
- **제목**: "Music Transformer" (Huang et al., 2018)
- **링크**: https://arxiv.org/abs/1809.04281
- **핵심**: Relative positional encoding

#### MusicVAE
- **제목**: "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music"
- **저자**: Roberts et al., 2018
- **링크**: https://arxiv.org/abs/1803.05428
- **핵심**: Hierarchical VAE for music

#### Jukebox
- **제목**: "Jukebox: A Generative Model for Music"
- **저자**: Dhariwal et al., 2020
- **링크**: https://arxiv.org/abs/2005.00341
- **핵심**: VQ-VAE for raw audio

#### MusicGen
- **제목**: "Simple and Controllable Music Generation"
- **저자**: Copet et al., 2023
- **링크**: https://arxiv.org/abs/2306.05284
- **핵심**: Text-to-music with EnCodec

### 책

#### 딥러닝
1. **"Deep Learning with PyTorch"**
   - Stevens, Antiga, Viehmann
   - 실전 PyTorch

2. **"Dive into Deep Learning"**
   - 링크: https://d2l.ai/
   - 무료, 인터랙티브

#### 음악 이론
1. **"The Jazz Theory Book"**
   - Mark Levine
   - 재즈 바이블

2. **"Charlie Parker Omnibook"**
   - Transcriptions
   - 악보 + 분석

---

## 👥 커뮤니티 & 네트워킹

### 한국 커뮤니티

#### 온라인
1. **AI Korea** (ai-korea.kr)
   - Slack 채널
   - 스터디, 세미나

2. **모두의 연구소** (modulabs.co.kr)
   - AI 연구 커뮤니티
   - 풀잎스쿨 (스터디)

3. **Facebook Groups**:
   - "Music & AI Korea"
   - "Deep Learning Korea"

#### 오프라인
1. **DEVIEW** (Naver)
   - 연 1회 컨퍼런스
   - AI 트랙

2. **PyTorch KR Meetup**
   - 분기별 모임

### 글로벌 커뮤니티

#### Reddit
1. **r/MachineLearning**
   - ML 전반
   - 논문 토론

2. **r/MusicInformationRetrieval**
   - 음악 AI 전문
   - 데이터셋, 논문

3. **r/MusicAI**
   - 음악 생성
   - 프로젝트 쇼케이스

#### Discord
1. **Hugging Face**
   - Transformers 커뮤니티
   - 빠른 답변

2. **AI Music Creation**
   - 음악 AI 전문
   - 콜라보레이션

3. **Eleuther AI**
   - 오픈소스 LLM
   - 고급 토론

### 학회 & 컨퍼런스

#### 음악 AI 전문
1. **ISMIR** (International Society for Music Information Retrieval)
   - 링크: https://ismir.net/
   - 연 1회, 논문 발표

2. **ICMC** (International Computer Music Conference)
   - 음악 + 기술
   - 실험적

#### AI 일반
1. **NeurIPS, ICML, ICLR**
   - 최고 수준 ML 학회
   - Workshop: Music & AI

2. **CVPR**
   - Computer Vision
   - 오디오 스펙트로그램 관련

---

## 🎓 취업 정보

### 한국 기업

#### AI 스타트업
1. **업스테이지** (Upstage)
   - LLM, 문서 AI
   - 채용: AI Engineer

2. **뤼튼** (Wrtn)
   - LLM 서비스
   - 채용: ML Engineer

3. **스캐터랩** (ScatterLab)
   - 대화 AI
   - 채용: Research Engineer

#### 음악 테크
1. **플로** (Flo)
   - 음악 스트리밍
   - 채용: 추천 시스템

2. **멜론**
   - 음악 플랫폼
   - 채용: Data Scientist

3. **뮤직카우**
   - 음악 투자
   - 채용: ML Engineer

#### 대기업
1. **네이버** (Clova AI)
   - 음성, 언어 AI
   - 채용: Research Engineer

2. **카카오** (카카오브레인)
   - 멀티모달 AI
   - 채용: AI Researcher

3. **LG AI연구원**
   - AI 전반
   - 채용: AI Scientist

### 해외 리모트

#### 음악 AI 스타트업
1. **Splice**
   - 음악 제작 도구
   - 리모트 가능

2. **AIVA**
   - AI 작곡
   - 유럽 기반

3. **Amper Music**
   - 배경음악 생성

#### 빅테크
1. **Google Magenta**
   - 연구 중심
   - 인턴십

2. **Meta AI**
   - MusicGen 팀
   - Full-time

---

## 📖 추가 리소스

### GitHub 레포지토리

1. **Magenta** (Google)
   - https://github.com/magenta/magenta
   - Music Transformer, MusicVAE 구현

2. **Music Transformer** (Official)
   - https://github.com/jason9693/MusicTransformer-pytorch
   - PyTorch 구현

3. **MusPy**
   - https://github.com/salu133445/muspy
   - 음악 데이터 처리 라이브러리

4. **pretty-midi**
   - https://github.com/craffel/pretty-midi
   - MIDI 처리 필수

### 블로그 & 튜토리얼

1. **Magenta Blog**
   - https://magenta.tensorflow.org/blog
   - 음악 AI 최신 연구

2. **Towards Data Science**
   - "Music Generation" 태그
   - 튜토리얼, 케이스 스터디

3. **Distill.pub**
   - https://distill.pub/
   - 인터랙티브 설명

---

## 🎯 체크리스트

### 환경 설정
- [ ] Google Colab 계정
- [ ] Kaggle 계정
- [ ] GitHub 계정
- [ ] Hugging Face 계정

### 필수 라이브러리 설치
- [ ] PyTorch
- [ ] pretty-midi
- [ ] librosa
- [ ] Transformers
- [ ] Gradio

### 데이터 수집
- [ ] Lakh MIDI Dataset 다운로드
- [ ] Charlie Parker MIDI 10개
- [ ] 재즈 오디오 샘플

### 커뮤니티 가입
- [ ] r/MachineLearning 구독
- [ ] AI Korea Slack 가입
- [ ] Discord 서버 가입

---

**"좋은 리소스는 학습 속도를 2배로 만듭니다. 적극 활용하세요!"** 🚀
