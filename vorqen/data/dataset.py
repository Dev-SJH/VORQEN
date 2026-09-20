"""
데이터셋 & 배치 생성
"""
import torch


def make_splits(data, train_ratio=0.9):
    """텐서를 train/val로 분할"""
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, device):
    """
    랜덤 위치에서 배치 추출

    Args:
        data: torch.LongTensor (1차원)
        block_size: 시퀀스 길이
        batch_size: 배치 크기
        device: 'cuda' / 'cpu'
    Returns:
        x, y: (B, T) 텐서. y는 x를 한 칸 shift한 것
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def load_text_file(path):
    """텍스트 파일 읽기"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()