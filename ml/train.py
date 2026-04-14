"""
한국 멸종위기 포유류 분류 모델 학습
- EfficientNet-B0 백본 + 커스텀 분류기 헤드
- 2단계 학습 전략: 백본 동결 → 전체 미세조정
실행: python train.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ═══════════════════════════════════════════════════════════════
#  하이퍼파라미터 설정 (여기 숫자를 바꿔가며 실험)
# ═══════════════════════════════════════════════════════════════
DATA_DIR         = "data/train"
MODEL_SAVE_PATH  = "../backend/app/model.pth"
LABELS_SAVE_PATH = "../backend/app/labels.json"

IMG_SIZE   = 224  # 입력 이미지 크기 (EfficientNet 기본 권장값)

# 1단계: 백본 동결, 헤드만 학습
PHASE1_EPOCHS = 5     # 실험값: 3 / 5 / 10
PHASE1_LR     = 1e-3  # 실험값: 1e-2 / 1e-3 / 1e-4

# 2단계: 전체 레이어 미세조정
PHASE2_EPOCHS = 20    # 실험값: 10 / 20 / 30
PHASE2_LR     = 1e-4  # 1단계보다 낮게 설정 (기존 특징 보존)

BATCH_SIZE   = 32     # 실험값: 16 / 32 / 64
WEIGHT_DECAY = 1e-4   # 과적합 방지 강도: 실험값 1e-3 / 1e-4 / 1e-5
DROPOUT1     = 0.4    # 첫 번째 드롭아웃 비율: 실험값 0.3 / 0.4 / 0.5
DROPOUT2     = 0.3    # 두 번째 드롭아웃 비율
HIDDEN_DIM   = 512    # 중간 레이어 크기: 실험값 256 / 512 / 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ═══════════════════════════════════════════════════════════════

print(f"디바이스: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── 데이터 증강 전략 ───────────────────────────────────────────
# 야생동물 사진 특성에 맞게 설계:
# - 좌우 반전: 동물은 어느 방향이든 같은 종
# - 색상 변화: 촬영 환경(밝기/계절)이 달라도 같은 종
# - 이동/회전: 사진 구도가 달라도 같은 종
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.05),  # 흑백 사진 대응
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── 데이터셋 로드 ──────────────────────────────────────────────
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_dataset   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"\n분류 종 ({num_classes}개): {class_names}")
print(f"전체 이미지 수: {len(train_dataset)}장")

# 8:2 비율로 학습/검증 분할
val_size   = int(len(train_dataset) * 0.2)
train_size = len(train_dataset) - val_size
train_idx, val_idx = random_split(
    range(len(train_dataset)), [train_size, val_size]
)

from torch.utils.data import Subset
train_ds = Subset(train_dataset, train_idx.indices)
val_ds   = Subset(val_dataset,   val_idx.indices)

print(f"학습: {len(train_ds)}장 / 검증: {len(val_ds)}장\n")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)


# ── 커스텀 분류기 헤드 설계 ────────────────────────────────────
#
#  EfficientNet-B0 특징 추출 (1280차원)
#          ↓
#   Dropout(0.4)        ← 과적합 방지
#          ↓
#   Linear(1280 → 512)  ← 고차원 특징을 압축
#          ↓
#   BatchNorm1d(512)    ← 학습 안정화
#          ↓
#   ReLU                ← 비선형 활성화
#          ↓
#   Dropout(0.3)        ← 추가 과적합 방지
#          ↓
#   Linear(512 → 14)    ← 최종 14종 분류
#
class CustomHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout1, dropout2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── 모델 구성 ──────────────────────────────────────────────────
backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = backbone.classifier[1].in_features
backbone.classifier = CustomHead(
    in_features, HIDDEN_DIM, num_classes, DROPOUT1, DROPOUT2
)
model = backbone.to(DEVICE)


# ── 검증 함수 ──────────────────────────────────────────────────
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, predicted = model(inputs).max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


# ── 에포크 학습 함수 ───────────────────────────────────────────
def run_epoch(optimizer, epoch, total_epochs, phase):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total
    val_acc   = evaluate()
    print(f"  [{phase}] Epoch {epoch:>2}/{total_epochs}  "
          f"Loss: {running_loss/len(train_loader):.4f}  "
          f"Train: {train_acc:.1f}%  Val: {val_acc:.1f}%",
          end="")
    return val_acc


criterion    = nn.CrossEntropyLoss()
best_val_acc = 0.0


# ════════════════════════════════════════════════════════════════
#  1단계: 백본 동결 — 커스텀 헤드만 학습
#  이유: ImageNet으로 학습된 특징 추출 능력을 유지하면서
#        새로운 분류기 헤드를 빠르게 방향 잡음
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"  1단계: 백본 동결 / 헤드만 학습 ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
print("=" * 60)

for param in model.features.parameters():
    param.requires_grad = False  # 백본 동결

optimizer1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
)
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)

for epoch in range(1, PHASE1_EPOCHS + 1):
    val_acc = run_epoch(optimizer1, epoch, PHASE1_EPOCHS, "1단계")
    scheduler1.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"model_state_dict": model.state_dict(),
                    "class_names": class_names, "num_classes": num_classes},
                   MODEL_SAVE_PATH)
        print("  ✅ 저장", end="")
    print()


# ════════════════════════════════════════════════════════════════
#  2단계: 백본 동결 해제 — 전체 미세조정
#  이유: 헤드가 안정화된 후 낮은 LR로 전체를 우리 데이터에 맞게 조정
#        LR을 낮게 써야 기존 특징이 망가지지 않음
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"  2단계: 전체 미세조정 ({PHASE2_EPOCHS} epochs, LR={PHASE2_LR})")
print("=" * 60)

for param in model.features.parameters():
    param.requires_grad = True  # 백본 동결 해제

optimizer2 = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS)

for epoch in range(1, PHASE2_EPOCHS + 1):
    val_acc = run_epoch(optimizer2, epoch, PHASE2_EPOCHS, "2단계")
    scheduler2.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"model_state_dict": model.state_dict(),
                    "class_names": class_names, "num_classes": num_classes},
                   MODEL_SAVE_PATH)
        print("  ✅ 저장", end="")
    print()


# ── 라벨 저장 ──────────────────────────────────────────────────
with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"  학습 완료!  최고 검증 정확도: {best_val_acc:.1f}%")
print(f"  모델 저장: {MODEL_SAVE_PATH}")
print("=" * 60)
