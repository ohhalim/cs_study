"""
Phase 2 - Audio Basics
librosa를 사용한 오디오 처리 및 특징 추출
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def create_sample_audio():
    """샘플 오디오 생성 (C major chord)"""
    print("🎵 Creating sample audio (C major chord)")
    print("=" * 50)

    sr = 22050  # 샘플링 레이트
    duration = 2.0  # 2초

    # C major chord: C4(262Hz), E4(330Hz), G4(392Hz)
    t = np.linspace(0, duration, int(sr * duration))

    # 3개 사인파 합성
    c4 = 0.3 * np.sin(2 * np.pi * 262 * t)
    e4 = 0.3 * np.sin(2 * np.pi * 330 * t)
    g4 = 0.3 * np.sin(2 * np.pi * 392 * t)

    chord = c4 + e4 + g4

    # 저장
    sf.write('sample_c_major_chord.wav', chord, sr)
    print(f"✅ Created: sample_c_major_chord.wav")
    print(f"   Duration: {duration} seconds")
    print(f"   Sample rate: {sr} Hz")
    print()

    return chord, sr


def load_and_analyze_audio(audio_path):
    """오디오 파일 로드 및 기본 분석"""
    print(f"\n{'='*50}")
    print(f"Loading: {audio_path}")
    print(f"{'='*50}\n")

    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=None)  # sr=None: 원본 샘플링 레이트 유지

    print("📊 Basic Information:")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {len(y) / sr:.2f} seconds")
    print(f"   Samples: {len(y):,}")
    print(f"   Channels: Mono")
    print(f"   Data type: {y.dtype}")
    print(f"   Min value: {y.min():.4f}")
    print(f"   Max value: {y.max():.4f}")
    print(f"   Mean: {y.mean():.4f}")
    print()

    return y, sr


def visualize_waveform(y, sr, output_path='waveform.png'):
    """Waveform 시각화"""
    print(f"🎨 Visualizing waveform")
    print("=" * 50)

    plt.figure(figsize=(14, 4))

    # Time axis
    time = np.arange(len(y)) / sr

    plt.plot(time, y, linewidth=0.5)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Waveform', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    print(f"✅ Saved: {output_path}")
    print()


def compute_spectrogram(y, sr):
    """Spectrogram 계산"""
    print(f"\n{'='*50}")
    print("Computing Spectrogram")
    print(f"{'='*50}\n")

    # STFT (Short-Time Fourier Transform)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    print(f"📊 Spectrogram:")
    print(f"   Shape: {S_db.shape}")
    print(f"   (Frequency bins x Time frames)")
    print(f"   Frequency bins: {S_db.shape[0]}")
    print(f"   Time frames: {S_db.shape[1]}")
    print()

    # 시각화
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.title('Spectrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=150)

    print(f"✅ Saved: spectrogram.png")
    print()

    return S_db


def compute_mel_spectrogram(y, sr):
    """Mel-spectrogram 계산 (음악 생성 AI의 핵심)"""
    print(f"\n{'='*50}")
    print("Computing Mel-Spectrogram")
    print(f"{'='*50}\n")

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    print(f"📊 Mel-Spectrogram:")
    print(f"   Shape: {mel_db.shape}")
    print(f"   (Mel bins x Time frames)")
    print(f"   Mel bins: {mel_db.shape[0]} (perceptually scaled)")
    print(f"   Time frames: {mel_db.shape[1]}")
    print()

    # 시각화
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Mel Frequency', fontsize=12)
    plt.title('Mel-Spectrogram (128 bins)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png', dpi=150)

    print(f"✅ Saved: mel_spectrogram.png")
    print()

    print("💡 Why Mel-scale?")
    print("   - 인간 청각은 선형이 아님 (낮은 주파수에 민감)")
    print("   - Mel scale: 1000Hz 이하는 선형, 이상은 로그")
    print("   - 음악 생성 모델 (MusicGen 등)은 Mel-spectrogram 사용")
    print()

    return mel_db


def compute_mfcc(y, sr):
    """MFCC (Mel-Frequency Cepstral Coefficients) 계산"""
    print(f"\n{'='*50}")
    print("Computing MFCC")
    print(f"{'='*50}\n")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    print(f"📊 MFCC:")
    print(f"   Shape: {mfcc.shape}")
    print(f"   (20 MFCC coefficients x Time frames)")
    print()

    # 시각화
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('MFCC Coefficients', fontsize=12)
    plt.title('MFCC (20 coefficients)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mfcc.png', dpi=150)

    print(f"✅ Saved: mfcc.png")
    print()

    print("💡 MFCC vs Mel-spectrogram:")
    print("   - MFCC: 음성 인식에 주로 사용 (압축된 표현)")
    print("   - Mel-spec: 음악 생성에 주로 사용 (더 많은 정보)")
    print()

    return mfcc


def compute_chroma(y, sr):
    """Chroma features (화음 분석)"""
    print(f"\n{'='*50}")
    print("Computing Chroma Features")
    print(f"{'='*50}\n")

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    print(f"📊 Chroma:")
    print(f"   Shape: {chroma.shape}")
    print(f"   (12 pitch classes x Time frames)")
    print(f"   Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B")
    print()

    # 시각화
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='plasma')
    plt.colorbar()
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Pitch Class', fontsize=12)
    plt.title('Chroma Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chroma.png', dpi=150)

    print(f"✅ Saved: chroma.png")
    print()

    print("💡 Chroma Features:")
    print("   - 옥타브 무관 (C4와 C5는 같은 C)")
    print("   - 화음 분석에 유용")
    print("   - Charlie Parker의 코드 진행 분석 가능")
    print()

    return chroma


def compute_tempo_and_beat(y, sr):
    """템포 및 비트 추정"""
    print(f"\n{'='*50}")
    print("Computing Tempo and Beats")
    print(f"{'='*50}\n")

    # 템포 추정
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    print(f"📊 Tempo:")
    print(f"   BPM: {tempo:.1f}")
    print(f"   Beats detected: {len(beats)}")
    print(f"   Beat frames: {beats[:10]}... (first 10)")
    print()

    # 비트 시간 변환
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"   Beat times (sec): {beat_times[:10]}... (first 10)")
    print()

    return tempo, beat_times


def pitch_shift_example(y, sr):
    """Pitch shifting (데이터 증강)"""
    print(f"\n{'='*50}")
    print("Pitch Shifting Example")
    print(f"{'='*50}\n")

    # +2 semitones (한 음 올림)
    y_shifted_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    # -3 semitones
    y_shifted_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)

    # 저장
    sf.write('shifted_up_2.wav', y_shifted_up, sr)
    sf.write('shifted_down_3.wav', y_shifted_down, sr)

    print(f"✅ Created:")
    print(f"   - shifted_up_2.wav (+2 semitones)")
    print(f"   - shifted_down_3.wav (-3 semitones)")
    print()

    print("💡 Data Augmentation:")
    print("   - Pitch shifting으로 데이터 증강")
    print("   - Charlie Parker 솔로를 여러 키로 변환")
    print("   - 학습 데이터 10배 확장 가능")
    print()


def time_stretch_example(y, sr):
    """Time stretching (템포 변경)"""
    print(f"\n{'='*50}")
    print("Time Stretching Example")
    print(f"{'='*50}\n")

    # 1.2배 빠르게
    y_faster = librosa.effects.time_stretch(y, rate=1.2)

    # 0.8배 느리게
    y_slower = librosa.effects.time_stretch(y, rate=0.8)

    # 저장
    sf.write('faster_1.2x.wav', y_faster, sr)
    sf.write('slower_0.8x.wav', y_slower, sr)

    print(f"✅ Created:")
    print(f"   - faster_1.2x.wav (1.2x speed)")
    print(f"   - slower_0.8x.wav (0.8x speed)")
    print()

    print("💡 Time Stretching:")
    print("   - 피치 변화 없이 템포만 변경")
    print("   - 재즈 연습: 느린 템포로 먼저 학습")
    print()


def main():
    """전체 오디오 처리 파이프라인"""
    print("\n" + "🎵" * 25)
    print(" " * 15 + "Audio Processing Tutorial")
    print("🎵" * 25 + "\n")

    # 1. 샘플 오디오 생성
    print("📝 Step 1: Create Sample Audio")
    y_sample, sr_sample = create_sample_audio()

    # 2. 오디오 로드
    print("📖 Step 2: Load Audio")
    y, sr = load_and_analyze_audio('sample_c_major_chord.wav')

    # 3. Waveform 시각화
    print("🎨 Step 3: Visualize Waveform")
    visualize_waveform(y, sr)

    # 4. Spectrogram
    print("🔍 Step 4: Spectrogram")
    S_db = compute_spectrogram(y, sr)

    # 5. Mel-spectrogram (중요!)
    print("🌟 Step 5: Mel-Spectrogram (Key for Music AI)")
    mel_db = compute_mel_spectrogram(y, sr)

    # 6. MFCC
    print("📊 Step 6: MFCC")
    mfcc = compute_mfcc(y, sr)

    # 7. Chroma
    print("🎼 Step 7: Chroma Features")
    chroma = compute_chroma(y, sr)

    # 8. Tempo & Beat
    print("⏱️ Step 8: Tempo and Beat")
    tempo, beats = compute_tempo_and_beat(y, sr)

    # 9. Pitch shifting
    print("🎚️ Step 9: Pitch Shifting")
    pitch_shift_example(y, sr)

    # 10. Time stretching
    print("⏩ Step 10: Time Stretching")
    time_stretch_example(y, sr)

    # 요약
    print("=" * 50)
    print("✅ All audio processing steps completed!")
    print("=" * 50)
    print("\n📁 Generated files:")
    print("   Audio files:")
    print("      - sample_c_major_chord.wav")
    print("      - shifted_up_2.wav")
    print("      - shifted_down_3.wav")
    print("      - faster_1.2x.wav")
    print("      - slower_0.8x.wav")
    print("   Visualizations:")
    print("      - waveform.png")
    print("      - spectrogram.png")
    print("      - mel_spectrogram.png")
    print("      - mfcc.png")
    print("      - chroma.png")
    print("\n💡 Next Steps:")
    print("   1. Listen to all audio files")
    print("   2. Compare visualizations")
    print("   3. Try with real jazz recordings")
    print("   4. Implement PyTorch Dataset with these features")
    print("\n🎷 Charlie Parker Connection:")
    print("   - Mel-spectrogram: 오디오 생성 모델 입력")
    print("   - Chroma: 코드 진행 분석")
    print("   - Pitch/Time: 데이터 증강으로 100+ 솔로 만들기")
    print()


if __name__ == "__main__":
    main()
