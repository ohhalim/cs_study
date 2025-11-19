# Phase 2: 오디오/MIDI 처리
## 음악 데이터를 AI가 이해하는 형태로 변환 (2개월)

---

## 🎯 목표

음악 데이터 (오디오, MIDI)를 딥러닝 모델이 학습할 수 있는 형태로 변환하는 기술을 마스터합니다.

### 완료 기준
- ✅ MIDI 파일을 파싱하고 분석 가능
- ✅ 오디오 파일에서 특징 추출 (Mel-spectrogram, MFCC, Chroma)
- ✅ 오디오 → MIDI 변환 가능
- ✅ Charlie Parker MIDI 데이터셋 100개 이상 수집 및 전처리
- ✅ 데이터 증강 (Pitch shift, Time stretch) 구현

---

## 📅 주차별 학습 계획

### Week 1-2: MIDI 처리 마스터
**목표**: MIDI 파일 완전 정복

#### MIDI 기초 이론
- MIDI 메시지 구조
- Note On/Off, Velocity, Timing
- Tempo, Time signature
- Track, Channel 개념

#### 핵심 라이브러리
1. **pretty_midi**: 가장 직관적인 MIDI 라이브러리
2. **mido**: Low-level MIDI 제어
3. **music21**: 음악 이론 분석

**실습**: `01_midi_basics.py`
```python
# MIDI 파일 읽기/쓰기
# Note 추출 및 통계 분석
# 피아노 롤 시각화
# MIDI → NumPy array 변환
```

#### Day 1-3: pretty_midi 마스터
- MIDI 파일 로드 및 탐색
- Note 정보 추출 (pitch, velocity, start, end)
- Instrument별 분리
- 피아노 롤 생성

#### Day 4-7: MIDI 전처리
- Quantization (박자 정렬)
- Transposition (조옮김)
- MIDI → Tensor 변환
- Batch processing

**실습**: `02_midi_preprocessing.py`
```python
# 100개 MIDI 파일 일괄 처리
# 데이터 정규화
# Train/Val/Test split
```

---

### Week 3-4: 오디오 처리 기초
**목표**: 오디오 신호를 이해하고 특징 추출

#### 오디오 신호 처리 이론
- Sampling rate, Bit depth
- Fourier Transform (FFT)
- Spectrogram, Mel-scale
- Window function, Hop length

#### 핵심 라이브러리
1. **librosa**: 음악 정보 검색 (MIR) 표준 라이브러리
2. **torchaudio**: PyTorch 통합 오디오 처리
3. **soundfile**: 오디오 I/O

**실습**: `03_audio_basics.py`
```python
# 오디오 파일 로드
# Waveform 시각화
# Spectrogram 계산 및 시각화
# Mel-spectrogram, MFCC 추출
```

#### Day 8-10: librosa로 특징 추출
- Waveform 로드 및 재생
- STFT (Short-Time Fourier Transform)
- Mel-spectrogram 계산
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma features (화음 분석)

#### Day 11-14: 오디오 전처리
- Resampling (44.1kHz → 16kHz)
- Normalization
- Silence removal
- Audio segmentation

**실습**: `04_audio_features.py`
```python
# Charlie Parker 녹음에서 특징 추출
# Feature 시각화
# Feature → Tensor 변환
```

---

### Week 5-6: 오디오 ↔ MIDI 변환
**목표**: 오디오와 MIDI를 자유자재로 변환

#### Audio → MIDI
- **Basic-Pitch** (Spotify 오픈소스):
  - 최신 딥러닝 기반 transcription
  - Polyphonic (화음) 지원
  - 높은 정확도

**실습**: `05_audio_to_midi.py`
```python
# Basic-Pitch로 Charlie Parker 솔로 변환
# 정확도 검증
# Post-processing (노이즈 제거)
```

#### MIDI → Audio
- **FluidSynth**: MIDI 렌더링
- **pretty_midi.fluidsynth()**: 간단한 변환
- Soundfont 선택 (악기 음색)

**실습**: `06_midi_to_audio.py`
```python
# MIDI → WAV 변환
# 다양한 악기로 렌더링
# 품질 평가
```

#### Day 15-21: Charlie Parker 데이터 수집
- YouTube에서 Charlie Parker 연주 수집
- 오디오 → MIDI 변환
- MIDI 정제 (에러 수정)
- 메타데이터 정리 (곡명, 템포, 키)

**목표**: 100개 이상 Charlie Parker 솔로 MIDI

---

### Week 7-8: 데이터 증강 & 파이프라인
**목표**: 데이터 증강으로 학습 데이터 10배 확장

#### 데이터 증강 기법
1. **Pitch Shift**: 조옮김 (-6 ~ +6 semitones)
2. **Time Stretch**: 템포 변화 (0.8x ~ 1.2x)
3. **Velocity Variation**: 다이나믹 변화
4. **Note Dropout**: 일부 음표 제거
5. **Rhythmic Variation**: 리듬 변형

**실습**: `07_data_augmentation.py`
```python
# 5가지 증강 기법 구현
# 1개 MIDI → 10개 변형 생성
# 음악적 자연스러움 유지
```

#### 데이터 파이프라인 구축
- 자동화된 전처리 파이프라인
- PyTorch Dataset 클래스
- DataLoader 통합
- 캐싱 및 성능 최적화

**실습**: `08_data_pipeline.py`
```python
# MusicDataset 클래스
# 실시간 데이터 증강
# Batch collation
```

---

## 💻 실습 프로젝트

### Project 1: MIDI 분석 도구
**난이도**: ⭐⭐☆☆☆

```python
# 기능:
- MIDI 파일 업로드
- Note 통계 (음역, 평균 velocity, 음표 수)
- 피아노 롤 시각화
- Chord progression 추출
```

**코드**: `projects/01_midi_analyzer.py`

---

### Project 2: 오디오 특징 추출기
**난이도**: ⭐⭐⭐☆☆

```python
# 기능:
- 오디오 파일 → Mel-spectrogram
- MFCC, Chroma 추출
- 비교 시각화
- CSV 저장
```

**코드**: `projects/02_audio_feature_extractor.py`

---

### Project 3: Charlie Parker 데이터셋 빌더
**난이도**: ⭐⭐⭐⭐☆

```python
# 목표: 100개 Charlie Parker MIDI 수집

# 단계:
1. YouTube에서 오디오 다운로드 (youtube-dl)
2. Audio → MIDI 변환 (Basic-Pitch)
3. MIDI 검증 및 정제
4. 메타데이터 정리
5. 데이터셋 구조화

# 폴더 구조:
charlie_parker_dataset/
├── raw_audio/
├── midi/
├── metadata.csv
└── processed/
```

**코드**: `projects/03_dataset_builder.py`

**음악 연결**: 이 데이터셋이 Phase 5의 핵심 재료!

---

### Project 4: 실시간 오디오 → MIDI 변환기
**난이도**: ⭐⭐⭐⭐☆

```python
# 기능:
- 마이크 입력
- 실시간 pitch detection
- MIDI 출력
- 악기 연주 → MIDI 기록
```

**코드**: `projects/04_realtime_transcription.py`

**응용**: 재즈 연습 도구, 즉흥 연주 분석

---

## 📚 Charlie Parker 데이터 수집 가이드

### 추천 곡 (솔로 위주)
1. **Ornithology** - 빠른 Be-bop, 코드 진행 명확
2. **Confirmation** - Rhythm changes, 전형적인 파커 스타일
3. **Ko-Ko** - Cherokee 코드, 기교적
4. **Anthropology** - I Got Rhythm 변형
5. **Billie's Bounce** - Blues, 초보자도 분석 가능
6. **Now's the Time** - F Blues, 반복 학습 좋음
7. **Scrapple from the Apple** - Honeysuckle Rose
8. **Yardbird Suite** - 멜로디컬한 솔로
9. **Donna Lee** - Indiana, 매우 빠름
10. **Au Privave** - F Blues 변형

### 데이터 소스
1. **YouTube**:
   - "Charlie Parker Ornithology solo"
   - "Charlie Parker transcription"
   - 고품질 녹음 우선

2. **MIDI 라이브러리**:
   - reddit.com/r/jazzmidi
   - freejazzlessons.com
   - jazzstandards.com

3. **전문 Transcription**:
   - Charlie Parker Omnibook (악보)
   - 악보 → MIDI 변환 (MuseScore)

### 품질 기준
- ✅ 명확한 솔로 구간
- ✅ 배경 소음 최소
- ✅ 템포 일정
- ✅ 최소 16초 이상
- ✅ 44.1kHz 이상 샘플링 레이트

---

## 🛠️ 도구 & 라이브러리

### 필수 설치
```bash
pip install pretty_midi
pip install mido
pip install music21
pip install librosa
pip install soundfile
pip install basic-pitch
pip install matplotlib
pip install seaborn
```

### 선택 설치
```bash
# FluidSynth (MIDI → Audio)
# Ubuntu/Debian
sudo apt-get install fluidsynth

# macOS
brew install fluid-synth

# MuseScore (악보 → MIDI)
# https://musescore.org/
```

---

## 📊 학습 진도 체크리스트

### Week 1-2: MIDI ✅
- [ ] pretty_midi로 MIDI 읽기/쓰기
- [ ] 피아노 롤 시각화
- [ ] MIDI → NumPy 변환
- [ ] 10개 MIDI 파일 전처리

### Week 3-4: Audio ✅
- [ ] librosa로 오디오 로드
- [ ] Mel-spectrogram 추출
- [ ] MFCC, Chroma 이해
- [ ] 오디오 전처리 파이프라인

### Week 5-6: Conversion ✅
- [ ] Basic-Pitch 설치 및 사용
- [ ] 5개 오디오 → MIDI 변환 성공
- [ ] MIDI → Audio 렌더링
- [ ] 50개 Charlie Parker MIDI 수집

### Week 7-8: Augmentation ✅
- [ ] 5가지 데이터 증강 구현
- [ ] PyTorch Dataset 클래스
- [ ] 전체 파이프라인 완성
- [ ] 최종 100+ MIDI 데이터셋

---

## 🎯 평가 기준

### 데이터 품질 (50%)
- [ ] Charlie Parker MIDI 100개 이상
- [ ] 메타데이터 정리 (곡명, BPM, 키)
- [ ] 정제된 데이터 (에러 없음)
- [ ] Train/Val/Test 분리

### 기술 역량 (30%)
- [ ] MIDI/Audio 자유자재로 처리
- [ ] 특징 추출 완벽 이해
- [ ] 데이터 증강 구현

### 파이프라인 (20%)
- [ ] 자동화된 전처리
- [ ] PyTorch 통합
- [ ] 재사용 가능한 코드

---

## 💡 실전 팁

### MIDI 처리
- **Quantization**: 재즈는 Swing feel이 중요! 과도한 quantization 주의
- **Velocity**: 파커의 다이나믹 특징 보존
- **Timing**: Syncopation이 핵심, 정확한 타이밍 중요

### 오디오 처리
- **샘플링 레이트**: 음악은 최소 22.05kHz, 가능하면 44.1kHz
- **Mel bins**: 128-256 (음악 생성), 80 (음성 인식)
- **Hop length**: 512 (22.05kHz), 256 (세밀한 분석)

### 데이터 수집
- **저작권 주의**: Charlie Parker는 공공 도메인 가능성 높음 (확인 필요)
- **다양성**: 여러 앨범, 시기에서 수집
- **일관성**: 동일한 전처리 파이프라인

---

## 🔗 다음 단계 연결

Phase 2를 완료하면:
- ✅ **MIDI 데이터**: 100개 Charlie Parker 솔로
- ✅ **전처리 능력**: 새로운 데이터도 즉시 처리
- ✅ **특징 추출**: 오디오 분석 기술 획득
- ✅ **파이프라인**: Phase 4 학습에 바로 사용

**➡️ [Phase 3: Music Theory & Jazz Analysis](../phase3-music-theory/learning-guide.md)**

이제 Charlie Parker의 음악적 패턴을 분석할 준비가 되었습니다!

---

## 📞 유용한 리소스

### 커뮤니티
- **r/MusicInformationRetrieval**: Reddit
- **ISMIR (학회)**: ismir.net
- **Magenta Discuss**: groups.google.com/g/magenta-discuss

### 튜토리얼
- **librosa Tutorials**: librosa.org/doc/latest/tutorial.html
- **Music21 User's Guide**: web.mit.edu/music21/
- **MIDI Basics**: midi.org/specifications

### 데이터셋
- **Lakh MIDI Dataset**: colin-raffel.com/projects/lmd/
- **MAESTRO**: magenta.tensorflow.org/datasets/maestro
- **Jazz MIDI**: reddit.com/r/jazzmidi

---

**"좋은 데이터는 좋은 모델의 시작입니다. Charlie Parker의 천재성을 데이터로 담아내세요."**

*Estimated Time: 60일 (하루 2-3시간)*
*Difficulty: ⭐⭐⭐☆☆*
*Next: Phase 3 - Music Theory* 🎼
