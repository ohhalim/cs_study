"""
Phase 1 - PyTorch Basics
Tensor 연산, Autograd, GPU 사용법 마스터
"""

import torch
import torch.nn as nn
import numpy as np
import time

def tensor_operations():
    """기본 Tensor 연산 연습"""
    print("="*50)
    print("1. Tensor 생성")
    print("="*50)

    # 다양한 방법으로 Tensor 생성
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.zeros(3, 4)
    c = torch.ones(2, 3, 4)
    d = torch.randn(2, 3)  # 정규분포
    e = torch.arange(0, 10, 2)

    print(f"1D Tensor: {a}")
    print(f"Zeros (3x4): \n{b}")
    print(f"Random (2x3): \n{d}")
    print(f"Range: {e}")

    # Tensor 타입 변환
    f = torch.tensor([1, 2, 3], dtype=torch.float32)
    g = f.long()  # int64로 변환
    print(f"\nFloat32: {f.dtype}, Long: {g.dtype}")

    print("\n" + "="*50)
    print("2. Tensor 연산")
    print("="*50)

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    # 기본 연산
    z1 = x + y
    z2 = torch.add(x, y)
    z3 = x * y  # element-wise
    z4 = torch.matmul(x, y.T)  # matrix multiplication

    print(f"Addition: {z1.shape}")
    print(f"Element-wise multiply: {z3.shape}")
    print(f"Matrix multiply: {z4.shape}")

    # Broadcasting
    a = torch.ones(3, 1)
    b = torch.ones(1, 4)
    c = a + b  # (3, 1) + (1, 4) = (3, 4)
    print(f"\nBroadcasting: {a.shape} + {b.shape} = {c.shape}")

    # Reshaping
    original = torch.randn(2, 3, 4)
    reshaped = original.view(2, 12)  # view는 메모리 공유
    reshaped_new = original.reshape(6, 4)  # reshape는 복사 가능

    print(f"\nReshape: {original.shape} -> {reshaped.shape}")


def autograd_basics():
    """자동 미분 (Autograd) 이해"""
    print("\n" + "="*50)
    print("3. Autograd - 자동 미분")
    print("="*50)

    # requires_grad=True로 gradient 추적 시작
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1

    print(f"x = {x.item()}")
    print(f"y = x^2 + 3x + 1 = {y.item()}")

    # 역전파
    y.backward()

    # dy/dx = 2x + 3 = 2*2 + 3 = 7
    print(f"dy/dx = {x.grad.item()} (예상: 7.0)")

    # 다변수 함수
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    c = a + b
    d = c.sum()

    d.backward()

    print(f"\na = {a.data}")
    print(f"b = {b.data}")
    print(f"d = sum(a + b) = {d.item()}")
    print(f"∂d/∂a = {a.grad}")
    print(f"∂d/∂b = {b.grad}")

    # Gradient 누적 방지
    a.grad.zero_()
    b.grad.zero_()
    print("\nGradient cleared!")


def neural_network_basics():
    """간단한 신경망 구현"""
    print("\n" + "="*50)
    print("4. Simple Neural Network")
    print("="*50)

    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # 모델 생성
    model = SimpleNet(10, 20, 5)
    print(model)

    # 파라미터 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Forward pass
    x = torch.randn(32, 10)  # batch_size=32, input_size=10
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 특정 레이어 파라미터 접근
    print(f"\nFirst layer weight shape: {model.fc1.weight.shape}")
    print(f"First layer bias shape: {model.fc1.bias.shape}")


def gpu_usage():
    """GPU 사용법 및 속도 비교"""
    print("\n" + "="*50)
    print("5. GPU Usage")
    print("="*50)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # CPU vs GPU 속도 비교
    size = 5000

    # CPU
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)

    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start

    print(f"\nCPU time: {cpu_time:.4f} seconds")

    # GPU
    if torch.cuda.is_available():
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)

        # Warm-up (GPU 초기화 시간 제외)
        _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()

        start = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()  # GPU 작업 완료 대기
        gpu_time = time.time() - start

        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # GPU 메모리 확인
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # 메모리 해제
        del x_gpu, y_gpu, z_gpu
        torch.cuda.empty_cache()
        print("GPU memory cleared!")


def practical_tips():
    """실전 팁"""
    print("\n" + "="*50)
    print("6. Practical Tips")
    print("="*50)

    # 1. Tensor와 NumPy 변환
    np_array = np.array([1, 2, 3, 4, 5])
    tensor = torch.from_numpy(np_array)
    back_to_np = tensor.numpy()

    print("1. NumPy ↔ Tensor 변환")
    print(f"   NumPy: {np_array}")
    print(f"   Tensor: {tensor}")

    # 2. no_grad() - Gradient 계산 안 함 (추론 시)
    x = torch.randn(10, requires_grad=True)

    with torch.no_grad():
        y = x * 2

    print(f"\n2. no_grad() context")
    print(f"   y.requires_grad = {y.requires_grad}")  # False

    # 3. detach() - Gradient graph에서 분리
    x = torch.randn(10, requires_grad=True)
    y = x.detach()

    print(f"\n3. detach()")
    print(f"   y.requires_grad = {y.requires_grad}")  # False

    # 4. in-place 연산 (메모리 효율적)
    x = torch.randn(3, 3)
    print(f"\n4. In-place operations")
    print(f"   Before: {id(x)}")
    x.add_(1)  # x = x + 1 (in-place)
    print(f"   After: {id(x)}")  # 같은 메모리 주소

    # 5. 모델 저장/로드
    model = nn.Linear(10, 5)

    # 저장
    torch.save(model.state_dict(), "model.pth")
    print(f"\n5. Model saved to model.pth")

    # 로드
    model_new = nn.Linear(10, 5)
    model_new.load_state_dict(torch.load("model.pth"))
    print(f"   Model loaded successfully!")

    # 6. 랜덤 시드 고정 (재현성)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"\n6. Random seed fixed for reproducibility")


def main():
    """모든 예제 실행"""
    print("\n" + "🔥"*25)
    print(" "*20 + "PyTorch Basics Tutorial")
    print("🔥"*25 + "\n")

    tensor_operations()
    autograd_basics()
    neural_network_basics()
    gpu_usage()
    practical_tips()

    print("\n" + "="*50)
    print("✅ All examples completed!")
    print("="*50)
    print("\n💡 Next Steps:")
    print("   1. Modify the code and experiment")
    print("   2. Try larger tensor sizes")
    print("   3. Move to 02_neural_network.py")
    print("\n")


if __name__ == "__main__":
    main()
