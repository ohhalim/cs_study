# Phase 3: 음악 이론 & 재즈 분석
## Charlie Parker 스타일의 수학적/이론적 이해 (1.5개월)

---

## 🎯 목표

Charlie Parker의 즉흥 연주를 **데이터와 이론으로 분석**하여 AI가 학습할 수 있는 패턴을 찾습니다.

### 완료 기준
- ✅ Be-bop 재즈 이론 이해 (코드, 스케일, 리듬)
- ✅ Charlie Parker 솔로 10개 이상 상세 분석
- ✅ 통계적 패턴 추출 (N-gram, Markov Chain)
- ✅ 음악 이론을 코드로 구현
- ✅ 재즈 코드 진행 데이터베이스 구축

---

## 📅 주차별 학습 계획

### Week 1-2: 재즈 이론 기초

#### 핵심 개념
1. **코드 (Chords)**
   - 기본 삼화음 (Major, Minor, Diminished, Augmented)
   - 7th chords (Maj7, Dom7, Min7, m7b5)
   - Extensions (9th, 11th, 13th)
   - Alterations (b9, #9, #11, b13)

2. **스케일 (Scales)**
   - Major, Minor (Natural, Harmonic, Melodic)
   - Blues scale
   - Bebop scale (Major, Dominant)
   - Altered scale, Whole tone, Diminished

3. **코드 진행 (Progressions)**
   - **ii-V-I**: 재즈의 DNA
   - **I Got Rhythm changes**: 많은 be-bop 곡의 기반
   - **Blues**: 12-bar blues, Bird Blues
   - **Rhythm changes**: AABA 형식

#### 실습
```python
# code/01_chord_theory.py
- 코드 구성음 생성
- 스케일 생성
- 코드-스케일 매칭
```

---

### Week 3-4: Be-bop 특징 분석

#### Charlie Parker의 핵심 기법
1. **Chromatic Approach Notes**
   - 반음 아래/위에서 접근
   - 코드 톤을 강조

2. **Enclosure**
   - 위/아래에서 목표 음 감싸기
   - 예: C를 B와 Db로 감싸기

3. **Bebop Scale**
   - 8음 스케일로 강박에 코드 톤
   - Passing tone 활용

4. **Rhythmic Displacement**
   - Syncopation (당김음)
   - 8분음표 중심의 빠른 프레이징

5. **Motivic Development**
   - 작은 모티프 변형 반복
   - Sequence (음정 이동)

#### Charlie Parker 대표곡 분석
**필수 분석 곡**:
1. **Ornithology** (How High the Moon)
2. **Confirmation** (Rhythm changes)
3. **Ko-Ko** (Cherokee)
4. **Billie's Bounce** (F Blues)
5. **Now's the Time** (F Blues)

**분석 항목**:
- 음역 분포
- 음정 간격 통계
- 리듬 패턴
- 코드 톤 vs 텐션 비율
- 프레이즈 길이

#### 실습
```python
# code/02_bebop_analysis.py
- Charlie Parker MIDI 통계 분석
- Chromatic approach 패턴 탐지
- 리듬 패턴 추출
```

---

### Week 5-6: 통계적 패턴 추출

#### N-gram 분석
- **Unigram**: 개별 음표 확률
- **Bigram**: 2개 음표 시퀀스
- **Trigram**: 3개 음표 시퀀스
- **4-gram**: 모티프 수준

```python
# 예시: Bigram
# F → G: 0.15
# F → A: 0.10
# F → Bb: 0.08
```

#### Markov Chain 모델
- 현재 상태에서 다음 상태 확률
- 1차 Markov: 이전 1개 음표
- 2차 Markov: 이전 2개 음표

```python
# code/03_markov_chain.py
- Charlie Parker 데이터로 Markov model 학습
- 간단한 멜로디 생성 (baseline)
```

#### 통계적 특징
- **Pitch entropy**: 음 다양성
- **Rhythm diversity**: 리듬 복잡도
- **Interval distribution**: 음정 간격
- **Chord tone ratio**: 코드 톤 비율

#### 실습
```python
# code/04_statistical_features.py
- 10개 Charlie Parker 솔로 통계 비교
- 다른 재즈 뮤지션과 비교 (Coltrane, Davis)
- Parker만의 특징 5가지 정량화
```

---

## 💻 실습 프로젝트

### Project 1: 재즈 코드 진행 생성기
**난이도**: ⭐⭐⭐☆☆

```python
# 기능:
- ii-V-I 진행 자동 생성
- Rhythm changes 템플릿
- Voice leading 적용
- MIDI 출력

# 예시:
generate_progression("ii-V-I", key="C", style="bebop")
# → Dm7 - G7 - Cmaj7
```

**코드**: `projects/01_chord_progression_generator.py`

---

### Project 2: Charlie Parker 패턴 데이터베이스
**난이도**: ⭐⭐⭐⭐☆

```python
# 목표: Parker의 lick(패턴) 자동 추출

# 단계:
1. 10개 솔로에서 반복되는 모티프 탐지
2. 코드 진행별로 분류 (ii-V, V-I, Turnaround 등)
3. Transposition으로 모든 키에 적용
4. 데이터베이스 구축 (JSON/SQLite)

# 결과:
{
  "ii-V-I": [
    {"pattern": [60, 62, 64, ...], "frequency": 15},
    ...
  ],
  "blues": [...]
}
```

**코드**: `projects/02_lick_database.py`

---

### Project 3: 재즈 즉흥연주 분석 도구
**난이도**: ⭐⭐⭐⭐☆

```python
# 기능:
- MIDI 업로드
- 자동 코드 진행 탐지
- Be-bop 기법 분석 (Chromatic approach, Enclosure 등)
- "Charlie Parker-ness" 점수 (0-100)
- 시각화 리포트

# 출력:
- Chord tone ratio: 65%
- Chromatic approaches: 42 instances
- Bebop scale usage: 78%
- Parker similarity: 73/100
```

**코드**: `projects/03_jazz_analyzer.py`

---

## 📚 학습 자료

### 재즈 이론 필수
1. **"The Jazz Theory Book"** - Mark Levine
2. **"Charlie Parker Omnibook"** - 악보 컬렉션
3. **"How to Improvise"** - Hal Crook

### 온라인 강의
1. **JazzGuitarLessons.net** - Chord theory
2. **Open Studio (YouTube)** - Be-bop analysis
3. **Rick Beato** - Music theory

### 논문 & 아티클
1. "Automatic Jazz Melody Generation" (Various papers)
2. "Statistical Modeling of Jazz Improvisation"
3. ISMIR papers on jazz analysis

---

## 📊 Charlie Parker 분석 결과 (예시)

### 음역
- **Range**: F3 (53) - C6 (84)
- **Most common**: G4 - D5 (67-74)
- **Tessitura**: 중상 음역 (재즈 색소폰 특징)

### 음정 간격
- **Steps (순차 진행)**: 55%
- **Leaps (도약)**: 45%
  - 3rd: 20%
  - 4th: 12%
  - 5th: 8%
  - Octave: 5%

### 리듬
- **8분음표**: 70%
- **16분음표**: 15%
- **4분음표**: 10%
- **Triplet**: 5%

### 코드 톤 비율
- **Chord tones**: 65%
- **Tensions (9, 11, 13)**: 20%
- **Chromatic/Passing**: 15%

---

## 🎼 Be-bop 코드 진행 데이터베이스

### 필수 진행
```
1. ii-V-I (Major)
   Dm7 - G7 - Cmaj7

2. ii-V-i (Minor)
   Dm7b5 - G7b9 - Cm7

3. I Got Rhythm (Bridge)
   | D7 | D7 | G7 | G7 |
   | C7 | C7 | F7 | F7 |

4. 12-bar Blues (F)
   | F7 | Bb7 | F7 | F7 |
   | Bb7 | Bb7 | F7 | F7 |
   | C7 | Bb7 | F7 | C7 |

5. Rhythm Changes (AABA)
   A: | Bb Gm7 | Cm7 F7 | Dm7 Gm7 | Cm7 F7 |
      | Fm7 Bb7 | Ebmaj7 Abmaj7 | Dm7 G7 | Cm7 F7 |
   B: | D7 | D7 | G7 | G7 |
      | C7 | C7 | F7 | F7 |
```

**코드**: `data/jazz_progressions.json`

---

## 💡 실전 팁

### 음악 이론 학습
- **DO**: 항상 악기로 소리 내보기 (피아노 추천)
- **DO**: 실제 녹음 들으며 분석
- **DON'T**: 이론만 외우지 말기

### 데이터 분석
- **DO**: 여러 곡 평균 내기 (일반화)
- **DO**: 시각화로 패턴 발견
- **DON'T**: 수치에만 의존, 음악성 중요

### Charlie Parker 연구
- **DO**: 여러 시기 녹음 비교
- **DO**: 다른 뮤지션과 차별화 포인트 찾기
- **DON'T**: 단순 모방, 창의성 이해가 목표

---

## 🔗 다음 단계 연결

Phase 3을 완료하면:
- ✅ **이론적 기반**: AI가 "왜" 그렇게 연주하는지 이해
- ✅ **패턴 데이터베이스**: 학습 데이터 augmentation에 활용
- ✅ **평가 지표**: 생성된 음악의 "Parker-ness" 측정 가능
- ✅ **통계 모델**: Baseline 비교용 Markov model

**➡️ [Phase 4: Music Generation Models](../phase4-model-training/learning-guide.md)**

이제 딥러닝으로 Charlie Parker 스타일을 학습할 준비 완료!

---

**"Charlie Parker는 규칙을 알았기에 규칙을 깰 수 있었습니다. AI도 마찬가지입니다."**

*Estimated Time: 45일 (하루 2시간)*
*Difficulty: ⭐⭐⭐⭐☆*
*Next: Phase 4 - Deep Learning Models* 🤖
