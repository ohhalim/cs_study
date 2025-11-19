"""
Phase 1 - Transformer Architecture
음악 생성의 핵심: Self-Attention과 Transformer 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)

        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Masking (for decoder or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Output projection
        output = self.W_o(attn_output)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Multi-head attention + residual + norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward + residual + norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class MusicTransformer(nn.Module):
    """
    Music Transformer for sequence generation
    입력: 음표 시퀀스 → 출력: 다음 음표 예측
    """

    def __init__(
        self,
        vocab_size,      # 음표 개수 (예: 128 MIDI notes)
        d_model=512,     # 모델 차원
        num_heads=8,     # Attention heads
        num_layers=6,    # Encoder layers
        d_ff=2048,       # Feed-forward 차원
        max_len=2048,    # 최대 시퀀스 길이
        dropout=0.1
    ):
        super(MusicTransformer, self).__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Causal mask for autoregressive generation
        (현재 위치 이후의 토큰을 볼 수 없게)

        Args:
            sz: sequence length

        Returns:
            mask: (sz, sz) lower triangular matrix
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask  # True는 볼 수 있음, False는 마스킹

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - token indices
            mask: (seq_len, seq_len) - attention mask

        Returns:
            output: (batch_size, seq_len, vocab_size) - logits
        """
        # Token embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Output projection
        output = self.fc_out(x)

        return output

    def generate(self, start_tokens, max_len=100, temperature=1.0):
        """
        Autoregressive generation

        Args:
            start_tokens: (batch_size, start_len) - 시작 토큰
            max_len: 생성할 최대 길이
            temperature: 샘플링 온도 (낮을수록 deterministic)

        Returns:
            generated: (batch_size, max_len) - 생성된 시퀀스
        """
        self.eval()
        device = start_tokens.device

        generated = start_tokens

        with torch.no_grad():
            for _ in range(max_len - start_tokens.size(1)):
                # Causal mask
                seq_len = generated.size(1)
                mask = self.generate_square_subsequent_mask(seq_len).to(device)

                # Forward pass
                logits = self.forward(generated, mask)  # (batch, seq_len, vocab_size)

                # 마지막 토큰의 logits
                next_token_logits = logits[:, -1, :] / temperature

                # Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

        return generated


def demo():
    """Transformer 데모"""
    print("\n" + "🎼"*25)
    print(" "*15 + "Music Transformer Demo")
    print("🎼"*25 + "\n")

    # ==================== 설정 ====================
    print("⚙️  Configuration")
    print("="*50)

    vocab_size = 128  # MIDI notes (0-127)
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 512
    batch_size = 4
    seq_len = 32

    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model dimension: {d_model}")
    print(f"   Attention heads: {num_heads}")
    print(f"   Encoder layers: {num_layers}")
    print(f"   Feed-forward dim: {d_ff}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print()

    # ==================== 모델 생성 ====================
    print("🧠 Building Model")
    print("="*50)

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print()

    # ==================== Forward Pass ====================
    print("🚀 Forward Pass")
    print("="*50)

    # 더미 입력 (MIDI note sequence)
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"   Input shape: {x.shape}")

    # Causal mask
    mask = model.generate_square_subsequent_mask(seq_len).to(device)
    print(f"   Mask shape: {mask.shape}")

    # Forward
    output = model(x, mask)
    print(f"   Output shape: {output.shape}")
    print(f"   Output: (batch_size, seq_len, vocab_size)")
    print()

    # ==================== Attention 시각화 ====================
    print("👀 Attention Visualization")
    print("="*50)

    # 첫 번째 encoder layer의 attention
    with torch.no_grad():
        x_embed = model.embedding(x) * math.sqrt(model.d_model)
        x_embed = model.pos_encoding(x_embed)

        attn_output, attn_weights = model.encoder_layers[0].self_attn(
            x_embed, x_embed, x_embed, None
        )

    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   (batch_size, num_heads, seq_len, seq_len)")

    # 첫 번째 샘플, 첫 번째 헤드의 attention
    first_attn = attn_weights[0, 0].cpu().numpy()
    print(f"\n   First sample, first head attention (5x5):")
    print(first_attn[:5, :5])
    print()

    # ==================== 음악 생성 ====================
    print("🎵 Music Generation")
    print("="*50)

    # 시작 토큰 (예: C major chord - C, E, G)
    start_tokens = torch.tensor([[60, 64, 67]]).to(device)  # MIDI notes
    print(f"   Start tokens: {start_tokens[0].tolist()} (C major chord)")

    generated = model.generate(start_tokens, max_len=20, temperature=1.0)
    print(f"   Generated sequence: {generated[0].tolist()}")
    print(f"   Length: {generated.size(1)}")
    print()

    # ==================== 학습 예시 ====================
    print("📚 Training Example")
    print("="*50)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 더미 학습 데이터
    train_x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    train_y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Forward
    mask = model.generate_square_subsequent_mask(seq_len).to(device)
    logits = model(train_x, mask)  # (batch, seq_len, vocab_size)

    # Loss 계산
    # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
    # targets: (batch, seq_len) -> (batch*seq_len)
    loss = criterion(logits.reshape(-1, vocab_size), train_y.reshape(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   Loss: {loss.item():.4f}")
    print()

    # ==================== 요약 ====================
    print("="*50)
    print("✅ Transformer Demo Completed!")
    print("="*50)
    print("\n💡 Key Concepts:")
    print("   1. Self-Attention: 시퀀스 내 모든 위치 간 관계 학습")
    print("   2. Multi-Head: 여러 관점에서 attention 계산")
    print("   3. Positional Encoding: 위치 정보 주입")
    print("   4. Causal Mask: 미래 정보 차단 (autoregressive)")
    print()
    print("🎼 Music AI Connection:")
    print("   - Music Transformer는 이 구조를 직접 사용")
    print("   - MIDI sequence → Token sequence")
    print("   - Autoregressive generation으로 음악 생성")
    print()


if __name__ == "__main__":
    # 재현성
    torch.manual_seed(42)

    demo()
