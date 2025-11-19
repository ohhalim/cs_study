# Phase 5: Charlie Parker AI 프로젝트
## BirdAI - 실전 Charlie Parker 스타일 AI 개발 (3개월)

---

## 🎯 최종 목표

**"BirdAI"**: 재즈 뮤지션이 Charlie Parker로 인정할 수 있는 즉흥 연주 AI 시스템 구축

### 성공 기준
- ✅ 블라인드 테스트 50% 이상 통과
- ✅ 코드 진행 기반 즉흥연주 가능
- ✅ 5분 이상 연속 생성 (반복 없이)
- ✅ Be-bop 특징 5가지 이상 구현
- ✅ 실시간 인터랙션 가능

---

## 🗺️ BirdAI 개발 로드맵

### Version 1.0 (Week 1-4): 기본 멜로디 생성
**목표**: 단순 MIDI 생성

```python
# 기능:
- Music Transformer 기반
- 무조건 멜로디 생성 (코드 무관)
- 32-bar 고정 길이

# 학습:
- Charlie Parker MIDI 100개
- 데이터 증강 → 1000개
- Epoch: 50

# 결과:
input: [시작 음표]
output: 32-bar 재즈 솔로 (MIDI)
```

---

### Version 2.0 (Week 5-8): 조건부 생성
**목표**: 코드 진행 따라 즉흥연주

```python
# 기능:
- Conditional generation
- 입력: 코드 진행 (예: Dm7 G7 Cmaj7)
- 출력: 코드에 맞는 솔로

# 구현:
class ConditionalTransformer(nn.Module):
    def __init__(self):
        self.chord_embedding = nn.Embedding(200, 512)  # 코드 임베딩
        self.note_embedding = nn.Embedding(128, 512)   # 음표 임베딩

    def forward(self, notes, chords):
        # 코드 + 음표 임베딩 결합
        chord_emb = self.chord_embedding(chords)
        note_emb = self.note_embedding(notes)
        combined = chord_emb + note_emb
        # Transformer...

# 학습:
- 코드 진행 자동 추출 (music21)
- <chord, melody> 쌍으로 학습

# 평가:
- ii-V-I 진행에 적절한 음 사용하는가?
- 코드 톤 비율 65% 이상
```

---

### Version 3.0 (Week 9-10): 실시간 인터랙션
**목표**: Call & Response

```python
# 기능:
- 사용자가 4-bar 멜로디 입력
- AI가 응답 4-bar 생성
- 재즈 잼 세션!

# 구현:
def call_and_response(user_input_midi):
    # 1. 입력 분석
    motif = extract_motif(user_input_midi)

    # 2. 유사 모티프 변형
    response = model.generate(
        context=user_input_midi,
        motif_constraint=motif,  # 모티프 재활용
        variation=True
    )

    return response

# 평가:
- 입력과 음악적 연결성
- 다양성 (단순 반복 아님)
```

---

### Version 4.0 (Week 11-12): 스타일 조절
**목표**: Parker-ness 조절 가능

```python
# 기능:
- Style intensity slider (0-100)
- 0: 보수적 (코드 톤 위주)
- 100: 매우 파커스러움 (Chromatic, 빠름)

# 구현:
# Conditional LayerNorm (FiLM)
class StyleConditionalLayer(nn.Module):
    def __init__(self):
        self.style_fc = nn.Linear(1, 512*2)  # gamma, beta

    def forward(self, x, style_intensity):
        gamma, beta = self.style_fc(style_intensity).chunk(2, dim=-1)
        return gamma * x + beta

# 또는 Classifier-Free Guidance
output = model(chords, style=None)  # Unconditional
output_styled = model(chords, style="parker")  # Conditional

final = (1 - guidance) * output + guidance * output_styled

# 평가:
- Style 0: 안전한 재즈
- Style 50: 균형잡힌 Parker
- Style 100: 매우 실험적
```

---

## 📁 프로젝트 구조

```
bird-ai/
├── data/
│   ├── charlie_parker/          # 100+ MIDI
│   ├── processed/               # 전처리 완료
│   └── augmented/               # 데이터 증강
│
├── models/
│   ├── music_transformer.py     # Core model
│   ├── conditional_transformer.py
│   ├── style_controller.py
│   └── vae_latent.py
│
├── training/
│   ├── train.py                 # 학습 스크립트
│   ├── dataset.py               # PyTorch Dataset
│   ├── config.yaml              # 하이퍼파라미터
│   └── utils.py
│
├── evaluation/
│   ├── metrics.py               # 정량 평가
│   ├── blind_test.py            # 블라인드 테스트
│   └── parker_score.py          # Parker-ness 점수
│
├── generation/
│   ├── generate.py              # MIDI 생성
│   ├── interactive.py           # 실시간 인터랙션
│   └── chord_following.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_evaluation.ipynb
│
└── README.md
```

---

## 🧠 핵심 기술 구현

### 1. Be-bop 특징 강화

```python
# Chromatic Approach 강제
class BebopLayer(nn.Module):
    def __init__(self):
        self.chromatic_attention = ChromaticAttention()

    def forward(self, x, target_note):
        # target_note 직전에 chromatic approach 선호
        approach_logits = self.chromatic_attention(x, target_note)
        return approach_logits

# 학습 시 Reward
if is_chromatic_approach(pred_note, target_note):
    reward += 1.0

loss = cross_entropy_loss - reward_weight * reward
```

### 2. 코드 톤 비율 유지

```python
def chord_tone_loss(predicted_notes, chord):
    """
    코드 톤 비율 65% 유지

    Args:
        predicted_notes: (batch, seq_len, vocab)
        chord: (batch, seq_len, chord_dim)
    """
    chord_tones = get_chord_tones(chord)  # [0, 4, 7] for C major

    # Predicted notes가 chord tone인지
    is_chord_tone = is_in(predicted_notes, chord_tones)

    # 목표: 65%
    actual_ratio = is_chord_tone.mean()
    target_ratio = 0.65

    loss = (actual_ratio - target_ratio) ** 2
    return loss

# Total loss
loss = ce_loss + 0.1 * chord_tone_loss + 0.05 * rhythm_loss
```

### 3. 리듬 다양성

```python
class RhythmGenerator(nn.Module):
    """
    음표 + 리듬 동시 생성

    Output: [(pitch, duration), ...]
    """
    def __init__(self):
        self.pitch_head = nn.Linear(512, 128)     # Pitch
        self.duration_head = nn.Linear(512, 32)   # Duration (quantized)

    def forward(self, x):
        pitch_logits = self.pitch_head(x)
        duration_logits = self.duration_head(x)

        return pitch_logits, duration_logits

# 재즈 리듬: 8분음표 70%, 16분음표 15%, ...
rhythm_prior = [0.7, 0.15, 0.05, 0.05, 0.05]  # 8th, 16th, quarter, ...

# KL divergence로 prior 따르게
rhythm_kl_loss = KL(predicted_rhythm, rhythm_prior)
```

---

## 📊 평가 시스템

### Parker-ness Score (0-100)

```python
def calculate_parker_score(generated_midi):
    """
    Charlie Parker 유사도 점수

    Returns:
        score: 0-100
        breakdown: dict of subscores
    """
    score = 0
    breakdown = {}

    # 1. 음역 (10점)
    pitch_range = get_pitch_range(generated_midi)
    if 53 <= pitch_range[0] and pitch_range[1] <= 84:  # F3-C6
        score += 10
    breakdown['pitch_range'] = 10

    # 2. 코드 톤 비율 (20점)
    chord_tone_ratio = calculate_chord_tone_ratio(generated_midi)
    if 0.60 <= chord_tone_ratio <= 0.70:
        score += 20
    elif 0.55 <= chord_tone_ratio <= 0.75:
        score += 15
    breakdown['chord_tone'] = 20

    # 3. Chromatic approach (15점)
    chromatic_count = count_chromatic_approaches(generated_midi)
    if chromatic_count >= 10:
        score += 15
    breakdown['chromatic'] = 15

    # 4. Bebop scale 사용 (15점)
    bebop_usage = calculate_bebop_scale_usage(generated_midi)
    if bebop_usage >= 0.5:
        score += 15
    breakdown['bebop'] = 15

    # 5. 리듬 다양성 (10점)
    rhythm_entropy = calculate_rhythm_entropy(generated_midi)
    if rhythm_entropy >= 1.5:
        score += 10
    breakdown['rhythm'] = 10

    # 6. 프레이즈 길이 (10점)
    phrase_lengths = detect_phrases(generated_midi)
    avg_phrase = np.mean(phrase_lengths)
    if 2 <= avg_phrase <= 4:  # 2-4 bar
        score += 10
    breakdown['phrase'] = 10

    # 7. 음정 간격 분포 (10점)
    interval_dist = calculate_interval_distribution(generated_midi)
    parker_dist = load_parker_interval_distribution()
    similarity = cosine_similarity(interval_dist, parker_dist)
    score += int(similarity * 10)
    breakdown['interval'] = 10

    # 8. Velocity 다양성 (10점)
    velocity_std = np.std([note.velocity for note in generated_midi.notes])
    if 15 <= velocity_std <= 25:
        score += 10
    breakdown['velocity'] = 10

    return score, breakdown

# 사용:
score, details = calculate_parker_score(generated_midi)
print(f"Parker-ness Score: {score}/100")
print(f"Details: {details}")

# 목표: 70/100 이상
```

### 블라인드 테스트

```python
# blind_test.py
import random

def blind_test(real_midis, generated_midis, num_testers=10):
    """
    재즈 뮤지션에게 블라인드 테스트

    Args:
        real_midis: Charlie Parker 진짜 솔로
        generated_midis: BirdAI 생성 솔로
        num_testers: 테스터 수

    Returns:
        success_rate: AI 솔로가 진짜로 인정받은 비율
    """
    # 50% Real, 50% Generated
    test_set = random.sample(real_midis, 10) + random.sample(generated_midis, 10)
    random.shuffle(test_set)

    # 테스터에게
    results = []
    for tester in range(num_testers):
        print(f"\nTester {tester + 1}:")
        correct = 0
        for idx, midi in enumerate(test_set):
            # MIDI 재생
            play_midi(midi)

            # 평가
            answer = input(f"#{idx + 1}: Is this Charlie Parker? (y/n): ")
            is_real = midi in real_midis

            if (answer == 'y' and is_real) or (answer == 'n' and not is_real):
                correct += 1

        accuracy = correct / len(test_set)
        results.append(accuracy)

    avg_accuracy = np.mean(results)
    print(f"\nAverage Accuracy: {avg_accuracy:.2%}")

    # AI 솔로가 진짜로 인정받은 비율
    fooled_rate = 1 - avg_accuracy  # 틀린 비율
    print(f"AI Fooled Rate: {fooled_rate:.2%}")

    return fooled_rate

# 목표: 50% 이상 (random guess)
```

---

## 🔧 트러블슈팅

### 문제 1: 모델이 반복만 함
**원인**: 과적합, 데이터 부족

**해결책**:
```python
# 1. Dropout 증가
model = MusicTransformer(dropout=0.3)  # 0.1 → 0.3

# 2. 데이터 증강 더 aggressive
augment_factor = 20  # 10 → 20

# 3. Nucleus sampling
generated = model.generate(top_p=0.9)  # Top-k 대신
```

### 문제 2: 코드 안 따름
**원인**: Conditioning 약함

**해결책**:
```python
# 1. Chord loss weight 증가
loss = ce_loss + 0.5 * chord_loss  # 0.1 → 0.5

# 2. Chord embedding 강화
self.chord_embedding = nn.Embedding(200, 1024)  # 512 → 1024
```

### 문제 3: 음악적으로 이상함
**원인**: 음악 이론 제약 없음

**해결책**:
```python
# Constrained decoding
def is_valid_note(prev_note, next_note, chord):
    # 1. 음역 제한
    if next_note < 53 or next_note > 84:
        return False

    # 2. 큰 도약 제한 (옥타브 이상)
    if abs(next_note - prev_note) > 12:
        return False

    # 3. 코드 외음은 passing note만
    if next_note not in get_scale(chord):
        # 이전/다음이 코드 톤이어야
        return is_passing_note(prev_note, next_note, next_next_note)

    return True

# 생성 시 filtering
logits[~is_valid_note(prev, next, chord)] = -inf
```

---

## 🎯 마일스톤

- [ ] **Week 4**: BirdAI v1.0 - 기본 생성 성공
- [ ] **Week 8**: BirdAI v2.0 - 코드 진행 따름
- [ ] **Week 10**: BirdAI v3.0 - 실시간 인터랙션
- [ ] **Week 12**: BirdAI v4.0 - 스타일 조절 가능
- [ ] **Finale**: Parker-ness Score 70+ 달성
- [ ] **Finale**: 블라인드 테스트 50%+ 통과

---

## 🔗 다음 단계

Phase 5 완료 시:
- ✅ **BirdAI v4.0**: 완성된 Charlie Parker AI
- ✅ **포트폴리오**: GitHub showcase 프로젝트
- ✅ **논문/블로그**: 기술 문서화
- ✅ **데모**: 웹 인터페이스 준비

**➡️ [Phase 6: Deployment & Portfolio](../phase6-deployment/learning-guide.md)**

이제 세상에 공개할 차례!

---

**"Bird lives through AI. Charlie Parker의 창의성을 코드로 영원히."**

*Estimated Time: 90일 (하루 4-5시간)*
*Difficulty: ⭐⭐⭐⭐⭐*
*Next: Phase 6 - Deployment* 🚀
