"""
Phase 3 - Markov Chain for Music Generation
Charlie Parker 데이터로 간단한 Markov model 구현 (Baseline)
"""

import numpy as np
import pretty_midi
from collections import defaultdict, Counter
import random


class MusicMarkovChain:
    """
    1차 Markov Chain for music generation
    P(note_t | note_t-1)
    """

    def __init__(self, order=1):
        """
        Args:
            order: Markov chain order (1=이전 1개 음표, 2=이전 2개 음표)
        """
        self.order = order
        self.transitions = defaultdict(Counter)
        self.start_notes = Counter()

    def train(self, note_sequences):
        """
        학습

        Args:
            note_sequences: List of note sequences (each is list of pitches)
        """
        for sequence in note_sequences:
            if len(sequence) < self.order + 1:
                continue

            # 시작 음표 (첫 order개)
            start_state = tuple(sequence[:self.order])
            self.start_notes[start_state] += 1

            # Transition 확률 학습
            for i in range(len(sequence) - self.order):
                current_state = tuple(sequence[i:i + self.order])
                next_note = sequence[i + self.order]
                self.transitions[current_state][next_note] += 1

        print(f"✅ Trained on {len(note_sequences)} sequences")
        print(f"   Unique states: {len(self.transitions)}")
        print(f"   Total transitions: {sum(sum(v.values()) for v in self.transitions.values())}")

    def generate(self, length=50, start_state=None, temperature=1.0):
        """
        멜로디 생성

        Args:
            length: 생성할 음표 개수
            start_state: 시작 상태 (None이면 랜덤)
            temperature: 샘플링 온도 (낮을수록 deterministic)

        Returns:
            generated: List of pitches
        """
        if start_state is None:
            # 랜덤 시작 상태
            start_state = random.choices(
                list(self.start_notes.keys()),
                weights=list(self.start_notes.values())
            )[0]

        generated = list(start_state)

        for _ in range(length - self.order):
            current_state = tuple(generated[-self.order:])

            if current_state not in self.transitions:
                # Unknown state → 랜덤 선택
                current_state = random.choice(list(self.transitions.keys()))

            # 다음 음표 확률
            next_notes = self.transitions[current_state]
            notes = list(next_notes.keys())
            counts = list(next_notes.values())

            # Temperature scaling
            if temperature != 1.0:
                counts = np.array(counts) ** (1.0 / temperature)

            # 정규화
            probs = counts / np.sum(counts)

            # 샘플링
            next_note = np.random.choice(notes, p=probs)
            generated.append(next_note)

        return generated

    def save_to_midi(self, pitches, output_path, tempo=120):
        """
        생성된 음표를 MIDI로 저장

        Args:
            pitches: List of MIDI pitches
            output_path: 저장 경로
            tempo: BPM
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        sax = pretty_midi.Instrument(program=65)  # Alto Sax

        # 간단한 리듬 (모두 8분음표)
        note_duration = 0.25  # 8분음표 (120BPM 기준)
        current_time = 0.0

        for pitch in pitches:
            note = pretty_midi.Note(
                velocity=90,
                pitch=int(pitch),
                start=current_time,
                end=current_time + note_duration
            )
            sax.notes.append(note)
            current_time += note_duration

        midi.instruments.append(sax)
        midi.write(output_path)
        print(f"✅ Saved: {output_path}")


def extract_note_sequences_from_midi(midi_files):
    """
    MIDI 파일에서 note sequence 추출

    Args:
        midi_files: List of MIDI file paths

    Returns:
        sequences: List of pitch sequences
    """
    sequences = []

    for midi_file in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(midi_file)

            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue

                # Note를 시간 순으로 정렬
                notes = sorted(instrument.notes, key=lambda x: x.start)

                # Pitch만 추출
                pitches = [note.pitch for note in notes]

                if len(pitches) > 10:  # 최소 10개 음표
                    sequences.append(pitches)

        except Exception as e:
            print(f"⚠️  Error loading {midi_file}: {e}")

    return sequences


def create_dummy_charlie_parker_data():
    """
    더미 Charlie Parker 스타일 데이터 생성

    특징:
    - 음역: F3 (53) - C6 (84)
    - Be-bop scale 중심
    - Chromatic approach
    """
    sequences = []

    # F Blues be-bop scale: F G Ab A Bb C D Eb F
    bebop_scale = [53, 55, 56, 57, 58, 60, 62, 63, 65,  # F3 octave
                   67, 69, 70, 72, 74, 75, 77,           # F4 octave
                   79, 81, 82, 84]                        # F5 octave

    for _ in range(100):  # 100개 시퀀스
        length = random.randint(20, 50)
        sequence = []

        for _ in range(length):
            # Be-bop scale에서 선택 (80%)
            if random.random() < 0.8:
                note = random.choice(bebop_scale)
            else:
                # Chromatic approach (20%)
                if len(sequence) > 0:
                    prev = sequence[-1]
                    # 반음 위/아래
                    note = prev + random.choice([-1, 1])
                    note = max(53, min(84, note))  # 범위 제한
                else:
                    note = random.choice(bebop_scale)

            sequence.append(note)

        sequences.append(sequence)

    return sequences


def analyze_transitions(markov_model, top_k=10):
    """Markov model의 transition 확률 분석"""
    print(f"\n{'='*50}")
    print("Transition Analysis")
    print(f"{'='*50}\n")

    # 가장 빈번한 transition
    all_transitions = []

    for state, next_notes in markov_model.transitions.items():
        for next_note, count in next_notes.items():
            all_transitions.append((state, next_note, count))

    # 정렬
    all_transitions.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {top_k} transitions:")
    print(f"{'State':<20} {'Next':<10} {'Count':<10}")
    print("-" * 40)

    for state, next_note, count in all_transitions[:top_k]:
        state_str = str(state)
        next_str = pretty_midi.note_number_to_name(next_note)
        print(f"{state_str:<20} {next_str:<10} {count:<10}")

    print()


def main():
    """Markov Chain 음악 생성 데모"""
    print("\n" + "🎵" * 25)
    print(" " * 10 + "Markov Chain Music Generation")
    print("🎵" * 25 + "\n")

    # 1. 데이터 생성 (실제로는 Charlie Parker MIDI 사용)
    print("📦 Step 1: Prepare Data")
    print("=" * 50)

    sequences = create_dummy_charlie_parker_data()
    print(f"   Generated {len(sequences)} Charlie Parker-style sequences")
    print(f"   Average length: {np.mean([len(s) for s in sequences]):.1f} notes")
    print()

    # 2. Markov model 학습
    print("🧠 Step 2: Train Markov Chain")
    print("=" * 50)

    # 1차 Markov
    markov_1 = MusicMarkovChain(order=1)
    markov_1.train(sequences)
    print()

    # 2차 Markov (더 복잡한 패턴)
    markov_2 = MusicMarkovChain(order=2)
    markov_2.train(sequences)
    print()

    # 3. Transition 분석
    print("📊 Step 3: Analyze Transitions")
    analyze_transitions(markov_1)

    # 4. 멜로디 생성
    print("🎼 Step 4: Generate Melodies")
    print("=" * 50)

    # 1차 Markov
    melody_1 = markov_1.generate(length=40, temperature=1.0)
    print(f"   1st-order Markov (length={len(melody_1)}):")
    print(f"   {melody_1[:20]}...")

    markov_1.save_to_midi(melody_1, 'markov_1st_order.mid')

    # 2차 Markov
    melody_2 = markov_2.generate(length=40, temperature=1.0)
    print(f"\n   2nd-order Markov (length={len(melody_2)}):")
    print(f"   {melody_2[:20]}...")

    markov_2.save_to_midi(melody_2, 'markov_2nd_order.mid')

    # Temperature 실험
    print(f"\n   Temperature experiments:")

    # Low temperature (더 predictable)
    melody_low = markov_1.generate(length=40, temperature=0.5)
    markov_1.save_to_midi(melody_low, 'markov_low_temp.mid')
    print(f"   - Low temp (0.5): More deterministic")

    # High temperature (더 random)
    melody_high = markov_1.generate(length=40, temperature=2.0)
    markov_1.save_to_midi(melody_high, 'markov_high_temp.mid')
    print(f"   - High temp (2.0): More random")

    print()

    # 5. 평가
    print("📈 Step 5: Evaluation")
    print("=" * 50)

    def calculate_diversity(pitches):
        """음의 다양성"""
        return len(set(pitches)) / len(pitches)

    def calculate_avg_interval(pitches):
        """평균 음정 간격"""
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        return np.mean(intervals)

    for name, melody in [("1st-order", melody_1),
                         ("2nd-order", melody_2),
                         ("Low temp", melody_low),
                         ("High temp", melody_high)]:
        diversity = calculate_diversity(melody)
        avg_interval = calculate_avg_interval(melody)
        print(f"   {name}:")
        print(f"      Diversity: {diversity:.2f}")
        print(f"      Avg interval: {avg_interval:.2f} semitones")

    print()

    # 요약
    print("=" * 50)
    print("✅ Markov Chain demonstration completed!")
    print("=" * 50)
    print("\n📁 Generated MIDI files:")
    print("   - markov_1st_order.mid")
    print("   - markov_2nd_order.mid")
    print("   - markov_low_temp.mid")
    print("   - markov_high_temp.mid")
    print("\n💡 Insights:")
    print("   - 2nd-order Markov: 더 일관된 패턴")
    print("   - Low temperature: 더 안전한 선택")
    print("   - High temperature: 더 창의적 (가끔 이상함)")
    print("\n🎯 Limitation of Markov Chain:")
    print("   - 짧은 패턴만 학습 (long-term structure 부족)")
    print("   - 리듬 정보 무시")
    print("   - 코드 진행 고려 안 함")
    print("\n   → 이것이 Transformer가 필요한 이유!")
    print()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    main()
