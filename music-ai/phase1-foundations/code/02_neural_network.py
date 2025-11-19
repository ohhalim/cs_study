"""
Phase 1 - Neural Network Training
완전한 학습 파이프라인: DataLoader, Loss, Optimizer, Training Loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class MusicPredictor(nn.Module):
    """간단한 다층 퍼셉트론 (MLP) - 음악 패턴 학습 시뮬레이션"""

    def __init__(self, input_size=88, hidden_sizes=[256, 128, 64], output_size=88):
        """
        Args:
            input_size: 입력 차원 (예: 88개 피아노 건반)
            hidden_sizes: 은닉층 크기 리스트
            output_size: 출력 차원 (다음 음표 예측)
        """
        super(MusicPredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 과적합 방지
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_dummy_music_data(num_samples=10000, input_size=88, output_size=88):
    """
    더미 음악 데이터 생성 (실제로는 MIDI에서 추출)

    시뮬레이션: 입력 음표 → 다음 음표 예측
    """
    # 음악적 패턴 시뮬레이션 (예: C major scale 편향)
    X = torch.randn(num_samples, input_size)

    # 간단한 규칙: 입력의 weighted sum + noise
    weights = torch.randn(input_size, output_size) * 0.1
    y = torch.matmul(X, weights) + torch.randn(num_samples, output_size) * 0.5

    # 활성화 확률로 변환 (0-1)
    y = torch.sigmoid(y)

    return X, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def plot_training_history(train_losses, val_losses):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("📊 Training curve saved to training_history.png")


def main():
    """전체 학습 파이프라인"""

    print("\n" + "🎵"*25)
    print(" "*15 + "Neural Network Training Pipeline")
    print("🎵"*25 + "\n")

    # ==================== 설정 ====================
    print("⚙️  Configuration")
    print("="*50)

    # 하이퍼파라미터
    config = {
        'input_size': 88,
        'hidden_sizes': [256, 128, 64],
        'output_size': 88,
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'train_split': 0.8,
    }

    for key, value in config.items():
        print(f"   {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}")
    print()

    # ==================== 데이터 준비 ====================
    print("📦 Preparing Data")
    print("="*50)

    X, y = create_dummy_music_data(num_samples=10000)
    print(f"   Dataset: {X.shape[0]} samples")
    print(f"   Input shape: {X.shape[1]}")
    print(f"   Output shape: {y.shape[1]}")

    # Train/Validation split
    train_size = int(config['train_split'] * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")

    # DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=0)

    print(f"   Batches per epoch: {len(train_loader)}")
    print()

    # ==================== 모델 생성 ====================
    print("🧠 Building Model")
    print("="*50)

    model = MusicPredictor(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        output_size=config['output_size']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    # ==================== Loss & Optimizer ====================
    print("🎯 Loss Function & Optimizer")
    print("="*50)

    criterion = nn.MSELoss()  # 회귀 문제
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print(f"   Criterion: {criterion}")
    print(f"   Optimizer: Adam (lr={config['learning_rate']})")
    print(f"   Scheduler: ReduceLROnPlateau")
    print()

    # ==================== 학습 ====================
    print("🚀 Training")
    print("="*50)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    # TensorBoard (선택)
    writer = SummaryWriter('runs/music_predictor')

    for epoch in range(config['num_epochs']):
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 검증
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # 로그
        print(f"Epoch [{epoch+1:3d}/{config['num_epochs']}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # TensorBoard 기록
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 모델 저장 (Best model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"   ✅ Best model saved! (Val Loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n   ⚠️  Early stopping triggered (patience={early_stop_patience})")
            break

    writer.close()
    print()

    # ==================== 결과 ====================
    print("📊 Results")
    print("="*50)
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final training loss: {train_losses[-1]:.4f}")
    print(f"   Total epochs: {len(train_losses)}")
    print()

    # 학습 곡선 시각화
    plot_training_history(train_losses, val_losses)

    # ==================== 모델 로드 & 추론 ====================
    print("🔮 Inference Example")
    print("="*50)

    # Best model 로드
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 샘플 추론
    with torch.no_grad():
        sample_input = X_val[:5].to(device)
        sample_output = model(sample_input)

        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {sample_output.shape}")
        print(f"   Sample prediction (first 10 dims):")
        print(f"   {sample_output[0, :10].cpu().numpy()}")

    print()
    print("="*50)
    print("✅ Training completed successfully!")
    print("="*50)
    print("\n📝 Next Steps:")
    print("   1. Visualize with TensorBoard: tensorboard --logdir=runs")
    print("   2. Experiment with hyperparameters")
    print("   3. Try different architectures (CNN, RNN)")
    print("   4. Move to 03_training_loop.py for advanced techniques")
    print()


if __name__ == "__main__":
    # 재현성을 위한 시드 고정
    torch.manual_seed(42)
    np.random.seed(42)

    main()
