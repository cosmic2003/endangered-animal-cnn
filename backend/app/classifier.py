# backend/app/classifier.py

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

# ── 종별 상세 정보 ─────────────────────────────────────────────
ANIMAL_INFO = {
    "산양": {
        "scientific_name": "Naemorhedus caudatus",
        "iucn_status": "VU (취약)",
        "description": "한국의 험준한 산악 지대에 서식하는 야생 염소의 일종입니다. 뛰어난 등반 능력을 가지고 있으며, 서식지 파괴와 밀렵으로 개체수가 급감했습니다.",
    },
    "반달가슴곰": {
        "scientific_name": "Ursus thibetanus",
        "iucn_status": "VU (취약)",
        "description": "가슴에 반달 모양의 흰 무늬가 특징인 곰으로, 지리산 일대에서 복원 사업이 진행 중입니다.",
    },
    "사향노루": {
        "scientific_name": "Moschus moschiferus",
        "iucn_status": "VU (취약)",
        "description": "수컷이 사향샘을 가진 소형 사슴류로, 사향 채취를 위한 남획으로 개체수가 크게 줄었습니다.",
    },
    "붉은박쥐": {
        "scientific_name": "Myotis rufoniger",
        "iucn_status": "EN (위기)",
        "description": "선명한 붉은 털이 특징인 박쥐로, 한국에서는 극히 희귀하며 동굴 서식지 파괴가 주요 위협입니다.",
    },
    "수달": {
        "scientific_name": "Lutra lutra",
        "iucn_status": "NT (준위협)",
        "description": "하천과 호수에 서식하는 반수생 포유류로, 수질 오염과 서식지 파괴로 개체수가 감소했습니다.",
    },
    "여우": {
        "scientific_name": "Vulpes vulpes",
        "iucn_status": "EN (위기, 한국 개체군)",
        "description": "한국에서는 거의 자취를 감춘 동물로, 소백산 일대에서 복원 사업이 진행 중입니다.",
    },
    "늑대": {
        "scientific_name": "Canis lupus",
        "iucn_status": "EX (한국 내 절멸)",
        "description": "한국에서는 이미 야생 절멸된 것으로 간주됩니다. 생태계 최상위 포식자로 중요한 역할을 합니다.",
    },
    "호랑이": {
        "scientific_name": "Panthera tigris",
        "iucn_status": "EN (위기)",
        "description": "한국의 상징 동물이지만 현재 야생에서는 절멸 상태입니다. 전 세계적으로도 3,900마리 미만만 생존합니다.",
    },
    "표범": {
        "scientific_name": "Panthera pardus",
        "iucn_status": "VU (취약)",
        "description": "한국 표범(아무르 표범)은 세계에서 가장 희귀한 대형 고양이과 동물 중 하나로, 야생 개체수는 100마리 미만입니다.",
    },
    "삵": {
        "scientific_name": "Prionailurus bengalensis",
        "iucn_status": "LC (관심 대상, 한국 내 취약)",
        "description": "고양이와 비슷하게 생긴 소형 야생 고양이로, 한국에서는 서식지 파괴로 개체수가 줄고 있습니다.",
    },
    "담비": {
        "scientific_name": "Martes flavigula",
        "iucn_status": "LC (관심 대상, 한국 내 감소)",
        "description": "노란 목 무늬가 특징인 족제비과 동물로, 산림 생태계의 중요한 포식자입니다.",
    },
    "하늘다람쥐": {
        "scientific_name": "Pteromys volans",
        "iucn_status": "LC (관심 대상, 한국 내 취약)",
        "description": "앞다리와 뒷다리 사이의 비막을 이용해 활공하는 다람쥐로, 노령 침엽수림에 의존합니다.",
    },
    "큰귀박쥐": {
        "scientific_name": "Plecotus auritus",
        "iucn_status": "LC (관심 대상, 한국 내 희귀)",
        "description": "몸집 대비 매우 큰 귀가 특징인 박쥐로, 동굴과 오래된 건물에 서식합니다.",
    },
    "긴꼬리딱새박쥐": {
        "scientific_name": "Miniopterus fuliginosus",
        "iucn_status": "NT (준위협)",
        "description": "긴 날개와 빠른 비행 속도가 특징인 박쥐로, 대규모 동굴 군집을 이루어 생활합니다.",
    },
}

# ── 이미지 전처리 ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── train.py와 동일한 커스텀 헤드 ────────────────────────────
class _CustomHead(nn.Module):
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


# ── 모델 로드 (서버 시작 시 1회만) ───────────────────────────
def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = _CustomHead(in_features, 512, num_classes, 0.4, 0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    print(f"[classifier] 모델 로드 완료 — {num_classes}종, device={DEVICE}")
    return model, class_names


_model, _class_names = _load_model()


# ── Grad-CAM 생성 ─────────────────────────────────────────────
def _compute_gradcam(original_image, tensor, class_idx):
    """EfficientNet-B0 마지막 특징층 기준 Grad-CAM 히트맵 생성"""
    target_layer = _model.features[-1]

    activations = [None]
    gradients   = [None]

    def fwd_hook(module, input, output):
        activations[0] = output

    def bwd_hook(module, grad_input, grad_output):
        gradients[0] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        _model.zero_grad()
        with torch.enable_grad():
            out   = _model(tensor)
            score = out[0, class_idx]
            score.backward()

        grads   = gradients[0]    # (1, C, h, w)
        acts    = activations[0]  # (1, C, h, w)
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1)).squeeze()
        cam     = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        # 224×224 으로 리사이즈
        cam_up = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        ) / 255.0

        # Jet 컬러맵 (matplotlib 없이 직접 구현)
        r = np.clip(1.5 - np.abs(4 * cam_up - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * cam_up - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * cam_up - 1), 0, 1)
        heatmap = np.stack([r, g, b], axis=-1)

        # 원본 이미지와 오버레이
        orig = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
        overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[Grad-CAM 오류] {e}")
        return None
    finally:
        h1.remove()
        h2.remove()


# ── 분류 함수 (main.py에서 호출) ──────────────────────────────
def classify_animal(image_bytes):
    if _model is None:
        return {"error": "모델 파일이 없습니다. ml/train.py를 먼저 실행하세요."}

    image  = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ── Top-3 예측 ──
    with torch.no_grad():
        outputs = _model(tensor)
        probs   = torch.softmax(outputs, dim=1)

    top3_probs, top3_idxs = probs[0].topk(3)
    top3 = [
        {"name": _class_names[idx.item()], "confidence": round(prob.item() * 100, 1)}
        for prob, idx in zip(top3_probs, top3_idxs)
    ]

    animal_name    = top3[0]["name"]
    confidence_val = top3[0]["confidence"]
    info           = ANIMAL_INFO.get(animal_name, {})

    # ── Grad-CAM ──
    gradcam_b64 = _compute_gradcam(image, tensor, top3_idxs[0].item())

    return {
        "animal_name":     animal_name,
        "scientific_name": info.get("scientific_name", ""),
        "iucn_status":     info.get("iucn_status", ""),
        "description":     info.get("description", ""),
        "confidence":      confidence_val,
        "top3":            top3,
        "gradcam":         gradcam_b64,
    }
