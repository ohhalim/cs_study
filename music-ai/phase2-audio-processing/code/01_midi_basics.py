"""
Phase 2 - MIDI Basics
pretty_midi를 사용한 MIDI 파일 처리
"""

import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_sample_midi():
    """샘플 MIDI 파일 생성 (C major scale)"""
    # MIDI 객체 생성
    midi = pretty_midi.PrettyMIDI()

    # Acoustic Grand Piano (Program 0)
    piano = pretty_midi.Instrument(program=0)

    # C major scale (C4-C5: 60-72)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    start_time = 0.0

    for pitch in notes:
        # Note(velocity, pitch, start, end)
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_time,
            end=start_time + 0.5
        )
        piano.notes.append(note)
        start_time += 0.5

    # Instrument를 MIDI에 추가
    midi.instruments.append(piano)

    # 저장
    midi.write('sample_c_major.mid')
    print("✅ Created: sample_c_major.mid")

    return midi


def read_midi_file(midi_path):
    """MIDI 파일 읽기 및 정보 추출"""
    print(f"\n{'='*50}")
    print(f"Reading: {midi_path}")
    print(f"{'='*50}\n")

    # MIDI 로드
    midi = pretty_midi.PrettyMIDI(midi_path)

    # 기본 정보
    print("📊 Basic Information:")
    print(f"   Duration: {midi.get_end_time():.2f} seconds")
    print(f"   Tempo: {midi.estimate_tempo():.1f} BPM")
    print(f"   Time signature: {midi.time_signature_changes}")
    print(f"   Number of instruments: {len(midi.instruments)}")
    print()

    # 각 악기별 정보
    for idx, instrument in enumerate(midi.instruments):
        print(f"🎹 Instrument {idx + 1}:")
        print(f"   Name: {instrument.name}")
        print(f"   Program: {instrument.program} ({pretty_midi.program_to_instrument_name(instrument.program)})")
        print(f"   Is drum: {instrument.is_drum}")
        print(f"   Number of notes: {len(instrument.notes)}")

        if len(instrument.notes) > 0:
            # Note 통계
            pitches = [note.pitch for note in instrument.notes]
            velocities = [note.velocity for note in instrument.notes]
            durations = [note.end - note.start for note in instrument.notes]

            print(f"   Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
            print(f"   Pitch range: {pretty_midi.note_number_to_name(min(pitches))} - "
                  f"{pretty_midi.note_number_to_name(max(pitches))}")
            print(f"   Average velocity: {np.mean(velocities):.1f}")
            print(f"   Average duration: {np.mean(durations):.3f} sec")

        print()

    return midi


def analyze_notes(midi):
    """Note 분석"""
    print(f"\n{'='*50}")
    print("Note Analysis")
    print(f"{'='*50}\n")

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # 드럼은 제외

        print(f"🎵 {instrument.name}:")

        # 처음 10개 note 출력
        print(f"\n   First 10 notes:")
        print(f"   {'Pitch':<10} {'Note':<8} {'Start':<10} {'End':<10} {'Duration':<10} {'Velocity':<10}")
        print(f"   {'-'*70}")

        for note in instrument.notes[:10]:
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start

            print(f"   {note.pitch:<10} {note_name:<8} {note.start:<10.3f} "
                  f"{note.end:<10.3f} {duration:<10.3f} {note.velocity:<10}")

        print()


def visualize_piano_roll(midi, output_path='piano_roll.png'):
    """피아노 롤 시각화"""
    print(f"\n{'='*50}")
    print("Visualizing Piano Roll")
    print(f"{'='*50}\n")

    # pretty_midi 내장 함수
    piano_roll = midi.get_piano_roll(fs=100)  # 100Hz sampling

    plt.figure(figsize=(14, 6))
    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Velocity')
    plt.xlabel('Time (100 Hz)', fontsize=12)
    plt.ylabel('MIDI Note Number', fontsize=12)
    plt.title('Piano Roll Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")

    # 악기별 피아노 롤
    fig, axes = plt.subplots(len(midi.instruments), 1,
                            figsize=(14, 3 * len(midi.instruments)))

    if len(midi.instruments) == 1:
        axes = [axes]

    for idx, instrument in enumerate(midi.instruments):
        # 개별 악기 MIDI 생성
        temp_midi = pretty_midi.PrettyMIDI()
        temp_midi.instruments.append(instrument)

        # Piano roll 생성
        piano_roll = temp_midi.get_piano_roll(fs=100)

        axes[idx].imshow(piano_roll, aspect='auto', origin='lower',
                        cmap='hot', interpolation='nearest')
        axes[idx].set_ylabel('MIDI Note', fontsize=10)
        axes[idx].set_title(f'{instrument.name}', fontsize=12, fontweight='bold')

    axes[-1].set_xlabel('Time (100 Hz)', fontsize=12)
    plt.tight_layout()
    plt.savefig('piano_roll_by_instrument.png', dpi=150)
    print(f"✅ Saved: piano_roll_by_instrument.png")


def midi_to_numpy(midi):
    """MIDI를 NumPy array로 변환 (딥러닝 입력)"""
    print(f"\n{'='*50}")
    print("Converting MIDI to NumPy")
    print(f"{'='*50}\n")

    # 방법 1: Piano roll (시간 x 피치)
    fs = 100  # 샘플링 주파수 (100Hz = 0.01초 해상도)
    piano_roll = midi.get_piano_roll(fs=fs)

    print(f"📊 Piano Roll:")
    print(f"   Shape: {piano_roll.shape}")
    print(f"   (128 MIDI notes x {piano_roll.shape[1]} time steps)")
    print(f"   Duration: {piano_roll.shape[1] / fs:.2f} seconds")
    print()

    # 방법 2: Note sequence (리스트)
    note_sequence = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            note_sequence.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })

    # 시작 시간으로 정렬
    note_sequence = sorted(note_sequence, key=lambda x: x['start'])

    print(f"📊 Note Sequence:")
    print(f"   Total notes: {len(note_sequence)}")
    print(f"   First 5 notes:")
    for note in note_sequence[:5]:
        print(f"      {note}")
    print()

    # 방법 3: One-hot encoding (시퀀스 모델용)
    # 각 time step에서 활성화된 note를 one-hot으로
    time_steps = int(midi.get_end_time() * fs)
    one_hot = np.zeros((time_steps, 128))

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            start_idx = int(note.start * fs)
            end_idx = int(note.end * fs)
            one_hot[start_idx:end_idx, note.pitch] = 1

    print(f"📊 One-Hot Encoding:")
    print(f"   Shape: {one_hot.shape}")
    print(f"   (Time steps x 128 MIDI notes)")
    print()

    return piano_roll, note_sequence, one_hot


def extract_melody(midi):
    """멜로디 라인 추출 (가장 높은 음)"""
    print(f"\n{'='*50}")
    print("Extracting Melody")
    print(f"{'='*50}\n")

    # Piano roll
    piano_roll = midi.get_piano_roll(fs=100)

    # 각 time step에서 가장 높은 음 (velocity > 0)
    melody = []

    for t in range(piano_roll.shape[1]):
        active_notes = np.where(piano_roll[:, t] > 0)[0]

        if len(active_notes) > 0:
            highest_note = active_notes[-1]  # 가장 높은 음
            melody.append(highest_note)
        else:
            melody.append(-1)  # 쉼표

    melody = np.array(melody)

    print(f"   Melody length: {len(melody)}")
    print(f"   Unique pitches: {len(np.unique(melody[melody >= 0]))}")
    print(f"   Melody (first 20): {melody[:20]}")
    print()

    return melody


def create_charlie_parker_style_phrase():
    """Charlie Parker 스타일 짧은 프레이즈 생성 (예시)"""
    print(f"\n{'='*50}")
    print("Creating Charlie Parker-Style Phrase")
    print(f"{'='*50}\n")

    midi = pretty_midi.PrettyMIDI()
    sax = pretty_midi.Instrument(program=65)  # Alto Sax

    # Be-bop 특징: 빠른 8분음표, Chromatic approach
    # F Blues: F Bb C7 F (간단히)
    phrase = [
        (60, 0.0, 0.25, 100),   # C
        (62, 0.25, 0.5, 90),    # D
        (64, 0.5, 0.75, 95),    # E
        (65, 0.75, 1.25, 110),  # F (강조)
        (67, 1.25, 1.5, 85),    # G
        (69, 1.5, 1.75, 90),    # A
        (70, 1.75, 2.0, 80),    # Bb (chromatic approach)
        (72, 2.0, 2.5, 105),    # C (옥타브 상승)
    ]

    for pitch, start, end, velocity in phrase:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        sax.notes.append(note)

    midi.instruments.append(sax)
    midi.write('parker_style_phrase.mid')

    print("✅ Created: parker_style_phrase.mid")
    print("   8 notes, Be-bop style phrase")
    print()

    return midi


def main():
    """전체 MIDI 처리 파이프라인"""
    print("\n" + "🎵" * 25)
    print(" " * 15 + "MIDI Processing Tutorial")
    print("🎵" * 25 + "\n")

    # 1. 샘플 MIDI 생성
    print("📝 Step 1: Create Sample MIDI")
    print("=" * 50)
    sample_midi = create_sample_midi()
    print()

    # 2. MIDI 파일 읽기
    print("📖 Step 2: Read MIDI File")
    midi = read_midi_file('sample_c_major.mid')

    # 3. Note 분석
    print("🔍 Step 3: Analyze Notes")
    analyze_notes(midi)

    # 4. 피아노 롤 시각화
    print("🎨 Step 4: Visualize Piano Roll")
    visualize_piano_roll(midi)

    # 5. NumPy 변환
    print("🔢 Step 5: Convert to NumPy")
    piano_roll, note_sequence, one_hot = midi_to_numpy(midi)

    # 6. 멜로디 추출
    print("🎼 Step 6: Extract Melody")
    melody = extract_melody(midi)

    # 7. Charlie Parker 스타일 프레이즈
    print("🎷 Step 7: Create Jazz Phrase")
    parker_midi = create_charlie_parker_style_phrase()

    # 요약
    print("=" * 50)
    print("✅ All MIDI processing steps completed!")
    print("=" * 50)
    print("\n📁 Generated files:")
    print("   - sample_c_major.mid")
    print("   - parker_style_phrase.mid")
    print("   - piano_roll.png")
    print("   - piano_roll_by_instrument.png")
    print("\n💡 Next Steps:")
    print("   1. Open MIDI files in MuseScore or GarageBand")
    print("   2. Experiment with different note patterns")
    print("   3. Try loading a real Charlie Parker MIDI")
    print("   4. Move to 02_midi_preprocessing.py")
    print()


if __name__ == "__main__":
    main()
