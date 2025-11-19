# Phase 6: 배포 & 포트폴리오
## BirdAI를 세상에 공개하고 취업하기 (1.5개월)

---

## 🎯 목표

BirdAI를 **실제 사용 가능한 서비스**로 배포하고, **취업을 위한 포트폴리오**를 완성합니다.

### 완료 기준
- ✅ 웹 데모 배포 (Gradio/Streamlit)
- ✅ API 서버 구축 (FastAPI)
- ✅ Spring Boot 통합 (Java + Python)
- ✅ GitHub 포트폴리오 완성
- ✅ 기술 블로그 3편 작성
- ✅ 이력서 업데이트

---

## 📅 주차별 계획

### Week 1-2: 웹 데모 개발

#### Gradio 데모 (추천!)
**장점**: 빠르고, ML 모델에 최적화

```python
# demo/gradio_app.py
import gradio as gr
from bird_ai import BirdAI

model = BirdAI.load_pretrained("checkpoints/best.pth")

def generate_jazz_solo(chord_progression, style_intensity, num_bars):
    """
    Args:
        chord_progression: "Dm7 G7 Cmaj7 A7" (text)
        style_intensity: 0-100 (slider)
        num_bars: 16/32/64 (dropdown)

    Returns:
        midi_file: Generated MIDI
        audio_file: Rendered audio (MP3)
        visualization: Piano roll image
    """
    # 1. Parse chords
    chords = parse_chord_progression(chord_progression)

    # 2. Generate
    midi = model.generate(
        chords=chords,
        style=style_intensity / 100.0,
        num_bars=num_bars
    )

    # 3. Render to audio
    audio = midi_to_audio(midi, soundfont="alto_sax.sf2")

    # 4. Visualize
    piano_roll_img = create_piano_roll(midi)

    # 5. Parker-ness score
    score, details = calculate_parker_score(midi)

    return midi, audio, piano_roll_img, f"Parker-ness: {score}/100"


# Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎷 BirdAI - Charlie Parker AI")
    gr.Markdown("Generate Be-bop jazz solos in the style of Charlie Parker")

    with gr.Row():
        with gr.Column():
            chord_input = gr.Textbox(
                label="Chord Progression",
                placeholder="Dm7 G7 Cmaj7 A7",
                value="Dm7 G7 Cmaj7"
            )
            style_slider = gr.Slider(
                0, 100,
                value=70,
                label="Parker-ness Intensity"
            )
            bars_dropdown = gr.Dropdown(
                [16, 32, 64],
                value=32,
                label="Number of Bars"
            )
            generate_btn = gr.Button("🎵 Generate Solo", variant="primary")

        with gr.Column():
            midi_output = gr.File(label="MIDI File")
            audio_output = gr.Audio(label="Audio Preview")
            piano_roll_output = gr.Image(label="Piano Roll")
            score_output = gr.Textbox(label="Analysis")

    generate_btn.click(
        fn=generate_jazz_solo,
        inputs=[chord_input, style_slider, bars_dropdown],
        outputs=[midi_output, audio_output, piano_roll_output, score_output]
    )

    # Examples
    gr.Examples(
        examples=[
            ["Dm7 G7 Cmaj7", 70, 32],
            ["F7 Bb7 F7 F7 Bb7 Bb7 F7 F7 C7 Bb7 F7 C7", 80, 64],  # F Blues
            ["Bbmaj7 Gm7 Cm7 F7", 60, 16],  # Rhythm changes
        ],
        inputs=[chord_input, style_slider, bars_dropdown]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Public URL 생성!
```

**배포**:
```bash
# Hugging Face Spaces (무료!)
# 1. https://huggingface.co/spaces 에서 new space
# 2. Upload app.py
# 3. 자동 배포!

# 결과: https://huggingface.co/spaces/your-name/bird-ai
```

---

### Week 3-4: FastAPI 서버

#### RESTful API
```python
# api/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from bird_ai import BirdAI

app = FastAPI(title="BirdAI API", version="1.0")
model = BirdAI.load_pretrained("checkpoints/best.pth")


class GenerateRequest(BaseModel):
    chord_progression: str
    style_intensity: float = 0.7
    num_bars: int = 32
    temperature: float = 0.9


@app.post("/generate")
async def generate(request: GenerateRequest):
    """재즈 솔로 생성"""
    midi = model.generate(
        chords=parse_chords(request.chord_progression),
        style=request.style_intensity,
        num_bars=request.num_bars,
        temperature=request.temperature
    )

    # MIDI 저장
    output_path = f"outputs/{uuid.uuid4()}.mid"
    midi.write(output_path)

    return {
        "midi_url": f"/download/{output_path}",
        "parker_score": calculate_parker_score(midi)[0]
    }


@app.post("/interactive")
async def interactive(user_midi: UploadFile = File(...)):
    """Call & Response"""
    # 사용자 MIDI 로드
    user_sequence = load_midi(user_midi.file)

    # AI 응답 생성
    response = model.call_and_response(user_sequence)

    # 결합
    combined = user_sequence + response

    output_path = f"outputs/response_{uuid.uuid4()}.mid"
    save_midi(combined, output_path)

    return FileResponse(output_path, media_type="audio/midi")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "BirdAI v4.0"}


# 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**배포**:
```bash
# Docker
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]

# Railway/Render (무료 티어)
railway login
railway init
railway up

# 결과: https://bird-ai.up.railway.app
```

---

### Week 5: Spring Boot 통합

#### Java ↔ Python 연동
```java
// spring-backend/src/main/java/com/birdai/service/AIService.java

@Service
public class AIService {

    @Value("${birdai.api.url}")
    private String apiUrl;  // http://localhost:8000

    private final RestTemplate restTemplate;

    public AIService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public GenerateResponse generateSolo(GenerateRequest request) {
        // FastAPI 호출
        String url = apiUrl + "/generate";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<GenerateRequest> entity = new HttpEntity<>(request, headers);

        GenerateResponse response = restTemplate.postForObject(
            url,
            entity,
            GenerateResponse.class
        );

        return response;
    }

    public byte[] downloadMidi(String midiUrl) {
        return restTemplate.getForObject(apiUrl + midiUrl, byte[].class);
    }
}

// Controller
@RestController
@RequestMapping("/api/jazz")
public class JazzController {

    private final AIService aiService;

    @PostMapping("/generate")
    public ResponseEntity<GenerateResponse> generate(@RequestBody GenerateRequest request) {
        GenerateResponse response = aiService.generateSolo(request);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/midi/{id}")
    public ResponseEntity<byte[]> downloadMidi(@PathVariable String id) {
        byte[] midi = aiService.downloadMidi("/download/" + id);

        return ResponseEntity.ok()
            .contentType(MediaType.parseMediaType("audio/midi"))
            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"solo.mid\"")
            .body(midi);
    }
}
```

**장점**:
- Spring Boot: 기업 표준 백엔드
- Python: AI 모델 (빠른 개발)
- **포트폴리오**: "Full-stack AI Developer" 증명!

---

### Week 6: 포트폴리오 완성

#### GitHub Repository
```
bird-ai/
├── README.md                    # ⭐ 핵심! 아래 참고
├── docs/
│   ├── architecture.md          # 시스템 구조
│   ├── training.md              # 학습 과정
│   └── evaluation.md            # 평가 결과
├── demo/
│   ├── gradio_app.py
│   └── screenshots/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   └── bird_ai/
├── tests/
├── requirements.txt
└── LICENSE
```

**README.md 템플릿**:
```markdown
# 🎷 BirdAI - Charlie Parker AI

Generate Be-bop jazz solos in the style of Charlie Parker using deep learning.

[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/your-name/bird-ai)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](link)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

![Demo GIF](demo/demo.gif)

## 🎯 Features

- **Conditional Generation**: Generate solos for any chord progression
- **Style Control**: Adjust Parker-ness intensity (0-100)
- **Interactive Mode**: Call & response jamming
- **High Quality**: 70+ Parker-ness score, 50%+ blind test pass rate

## 🚀 Quick Start

```bash
pip install bird-ai
```

```python
from bird_ai import BirdAI

model = BirdAI.from_pretrained("bird-ai/parker-v4")
midi = model.generate("Dm7 G7 Cmaj7", style=0.7)
midi.save("solo.mid")
```

## 📊 Results

| Metric | Score |
|--------|-------|
| Parker-ness Score | 73/100 |
| Blind Test | 52% |
| Chord Tone Ratio | 67% |

## 🏗️ Architecture

- **Model**: Music Transformer (6 layers, 512 dim)
- **Training Data**: 100+ Charlie Parker solos (augmented to 1000+)
- **Conditioning**: Chord progression embedding
- **Style Control**: FiLM layers

## 📝 Blog Posts

1. [How I Built Charlie Parker AI](link)
2. [Music Transformer from Scratch](link)
3. [Evaluating Jazz AI: Beyond Accuracy](link)

## 📄 License

MIT License. Free for personal and commercial use.

## 🙏 Acknowledgments

- Charlie Parker for the inspiration
- Magenta team for Music Transformer
- Jazz community for feedback

---

*"Bird lives through AI"* 🐦
```

---

#### 기술 블로그 3편

**블로그 1**: "Charlie Parker AI를 만들며 배운 것들"
- 동기 (왜 음악 AI를 시작했는지)
- 여정 (12개월 로드맵)
- 어려웠던 점 & 해결책
- 결과 및 데모

**블로그 2**: "Music Transformer 처음부터 구현하기"
- Transformer 리뷰
- Music-specific modifications
- PyTorch 코드 (핵심 부분)
- 학습 팁 & 트러블슈팅

**블로그 3**: "재즈 AI 평가하기: 정량적 vs 정성적"
- 음악 생성 평가의 어려움
- Parker-ness score 설계
- 블라인드 테스트 결과
- 재즈 뮤지션 피드백

**플랫폼**:
- Medium (영어)
- Velog/Tistory (한국어)
- dev.to

---

## 💼 취업 전략

### 이력서 업데이트

```
[프로젝트]

BirdAI - Charlie Parker 스타일 재즈 AI 생성 시스템  (2025.06 - 2025.12)

• Music Transformer 기반 조건부 음악 생성 모델 설계 및 학습 (PyTorch)
• 100+ Charlie Parker MIDI 데이터 수집, 전처리 및 증강 (pretty_midi, librosa)
• 코드 진행 기반 즉흥연주 생성 (Conditional generation)
• Parker-ness 평가 지표 개발 및 블라인드 테스트 수행 (52% 통과율)
• Gradio 웹 데모 및 FastAPI 서버 구축, Hugging Face Spaces 배포
• Spring Boot와 Python AI 서버 통합 (RESTful API, Docker)

기술 스택: Python, PyTorch, Transformers, FastAPI, Gradio, Spring Boot, Docker
성과: GitHub Star 50+, Hugging Face Demo 1000+ 사용

[기술 블로그]

• "Charlie Parker AI 개발기" - Medium (조회수 5,000+)
• "Music Transformer 구현 가이드" - Velog
• "재즈 AI 평가 방법론" - dev.to
```

### 목표 기업

**스타트업**:
- 음악 테크 (뮤직카우, 플로, 멜론)
- AI 스타트업 (업스테이지, 스캐터랩, 뤼튼)
- 엔터테인먼트 AI

**대기업**:
- 네이버 (Clova AI)
- 카카오 (카카오브레인)
- LG AI연구원

**해외 리모트**:
- Splice, Soundtrap
- AI music startups

### 포트폴리오 피칭

**엘리베이터 피치** (30초):
> "저는 Charlie Parker 스타일의 재즈 즉흥연주를 생성하는 AI, BirdAI를 개발했습니다.
> Music Transformer를 기반으로 코드 진행에 맞는 재즈 솔로를 실시간 생성하며,
> 블라인드 테스트에서 52%의 재즈 뮤지션이 진짜 Charlie Parker로 인정했습니다.
> 웹 데모는 Hugging Face에 배포되어 있으며, Spring Boot와 통합하여
> 프로덕션 레벨 AI 서비스 경험도 갖추었습니다."

---

## 📊 성과 지표

### GitHub
- [ ] README 완성도 90% 이상
- [ ] Star 10개 이상
- [ ] Fork 5개 이상
- [ ] 코드 문서화 80% 이상

### 데모
- [ ] Hugging Face Spaces 배포
- [ ] 50+ 사용자 테스트
- [ ] 피드백 수집 및 개선

### 블로그
- [ ] 3편 이상 작성
- [ ] 총 조회수 1,000+ (합계)
- [ ] 댓글/피드백 10+ 개

### 네트워킹
- [ ] LinkedIn 프로필 업데이트
- [ ] AI/음악 커뮤니티 활동
- [ ] 컨퍼런스 발표 (선택)

---

## 🔗 최종 점검

- ✅ BirdAI v4.0 완성
- ✅ 웹 데모 배포
- ✅ API 서버 구축
- ✅ Spring Boot 통합
- ✅ GitHub 포트폴리오
- ✅ 기술 블로그 3편
- ✅ 이력서 업데이트
- ✅ 취업 준비 완료!

---

**"12개월의 여정을 마치며, 당신은 이제 음악 AI 엔지니어입니다."**

*Estimated Time: 45일*
*Difficulty: ⭐⭐⭐☆☆*
*Next: 취업! 그리고 새로운 프로젝트* 🎉

---

## 🎓 더 나아가기 (선택)

1. **논문 작성**: ISMIR, ICMC 학회 제출
2. **오픈소스**: PyPI 패키지 배포
3. **비즈니스**: 재즈 교육 도구로 상용화
4. **확장**: 다른 뮤지션 (Coltrane, Davis, ...)
5. **실시간 잼**: VST Plugin 개발

**"지금은 끝이 아니라 시작입니다."** 🚀
