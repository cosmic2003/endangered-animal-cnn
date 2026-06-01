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
        "iucn_status": "LC (관심 대상, 한국 내 야생 절멸)",
        "description": "전 세계적으로는 관심 대상 등급이나, 한국에서는 이미 야생 절멸된 것으로 간주됩니다. 생태계 최상위 포식자로 중요한 역할을 합니다.",
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
    # ── 신규 추가: 포유류 ──────────────────────────────────────
    "스라소니": {
        "scientific_name": "Lynx lynx",
        "iucn_status": "LC (관심 대상, 한국 내 야생 절멸 추정)",
        "description": "귀 끝의 검은 붓털과 짧은 꼬리가 특징인 중형 고양이과 동물입니다. 전 세계적으로는 안정적이나 한국에서는 이미 야생 절멸한 것으로 추정됩니다.",
    },
    "대륙사슴": {
        "scientific_name": "Cervus nippon",
        "iucn_status": "LC (관심 대상, 한국 자생 개체군 절멸)",
        "description": "몸의 흰 반점 때문에 '꽃사슴'이라 불리는 사슴으로, 한국 자생 개체군은 절멸했고 현재 복원·사육 개체가 남아 있습니다.",
    },
    # ── 신규 추가: 파충류·양서류 ───────────────────────────────
    "구렁이": {
        "scientific_name": "Elaphe schrenckii",
        "iucn_status": "멸종위기 야생생물 II급 (천연기념물)",
        "description": "한국에서 가장 큰 뱀으로 독이 없으며, 쥐 등을 잡아먹어 생태계 균형에 기여합니다. 서식지 파괴와 포획으로 개체수가 크게 줄었습니다.",
    },
    "남생이": {
        "scientific_name": "Mauremys reevesii",
        "iucn_status": "EN (위기) · 멸종위기 II급",
        "description": "한국 토착 민물거북으로, 등딱지에 세 줄의 융기가 있습니다. 외래종 붉은귀거북과의 경쟁과 서식지 훼손으로 위협받고 있습니다.",
    },
    "표범장지뱀": {
        "scientific_name": "Eremias argus",
        "iucn_status": "LC (관심 대상) · 멸종위기 II급",
        "description": "몸에 표범 무늬가 있는 도마뱀으로, 해안 사구와 강가 모래밭에 서식합니다. 해안 개발로 서식지가 급격히 줄고 있습니다.",
    },
    "맹꽁이": {
        "scientific_name": "Kaloula borealis",
        "iucn_status": "LC (관심 대상) · 멸종위기 II급",
        "description": "장마철에 '맹꽁' 하고 우는 둥근 몸의 땅속 개구리로, 도시 개발로 산란 습지가 사라지며 위협받고 있습니다.",
    },
    "금개구리": {
        "scientific_name": "Pelophylax chosenicus",
        "iucn_status": "VU (취약) · 멸종위기 II급 · 한국 고유종",
        "description": "등에 금색 융기선 두 줄이 뚜렷한 한국 고유 개구리로, 논·습지 감소와 외래종 유입으로 개체수가 줄고 있습니다.",
    },
    # ── 신규 추가: 해양 포유류 ────────────────────────────────────
    "점박이물범": {
        "scientific_name": "Phoca largha",
        "iucn_status": "LC (관심 대상) · 멸종위기 II급",
        "description": "백령도와 서해안에 서식하는 물범으로, 얼룩덜룩한 점박이 무늬가 특징입니다. 중국·러시아 등 번식지 환경 변화와 혼획으로 위협받고 있습니다.",
    },
    "상괭이": {
        "scientific_name": "Neophocaena asiaeorientalis",
        "iucn_status": "VU (취약) · 멸종위기 II급",
        "description": "등지느러미가 없는 소형 돌고래로, '웃는 돌고래'라는 별명을 가집니다. 서·남해안과 한강에도 출몰하며 혼획과 수질 오염으로 개체수가 급감하고 있습니다.",
    },
    # ── 신규 추가: 국제 멸종위기종 (동물원) ─────────────────────
    "자이언트판다": {
        "scientific_name": "Ailuropoda melanoleuca",
        "iucn_status": "VU (취약)",
        "description": "흑백 털과 특유의 눈 무늬로 세계에서 가장 사랑받는 동물 중 하나입니다. 중국 쓰촨성 대나무 숲이 주 서식지이며, 집중적인 보전 활동으로 개체수가 서서히 회복되고 있습니다.",
    },
    "눈표범": {
        "scientific_name": "Panthera uncia",
        "iucn_status": "VU (취약)",
        "description": "중앙아시아 고산 지대에 사는 대형 고양이과 동물로, 두꺼운 꼬리와 연기 같은 회색 얼룩무늬가 특징입니다. '설산의 유령'이라는 별명처럼 매우 은밀하게 생활합니다.",
    },
    "북극곰": {
        "scientific_name": "Ursus maritimus",
        "iucn_status": "VU (취약)",
        "description": "세계 최대의 육상 육식 동물로, 북극해 해빙에 의존해 살아갑니다. 기후변화로 해빙이 줄어들면서 서식지가 크게 위협받고 있습니다.",
    },
    "아시아코끼리": {
        "scientific_name": "Elephas maximus",
        "iucn_status": "EN (위기)",
        "description": "아프리카코끼리보다 귀가 작고 등이 둥근 편입니다. 삼림 파괴와 밀렵, 인간과의 충돌로 개체수가 5만 마리 이하로 줄었습니다.",
    },
    "오랑우탄": {
        "scientific_name": "Pongo pygmaeus",
        "iucn_status": "CR (위급)",
        "description": "보르네오섬에 사는 대형 유인원으로, 긴 팔과 붉은 털이 특징입니다. 팜유 농장 개발로 인한 삼림 파괴가 가장 큰 위협 요인입니다.",
    },
    "고릴라": {
        "scientific_name": "Gorilla gorilla",
        "iucn_status": "CR (위급)",
        "description": "지구상에서 가장 큰 영장류로, 온화한 성격과 높은 지능을 가집니다. 서부 저지대 고릴라는 밀렵과 에볼라 바이러스, 서식지 파괴로 개체수가 급감했습니다.",
    },
    "기린": {
        "scientific_name": "Giraffa camelopardalis",
        "iucn_status": "VU (취약)",
        "description": "지구상에서 가장 키가 큰 동물로, 특유의 긴 목과 독특한 얼룩무늬가 특징입니다. 지난 30년간 아프리카 개체수가 40% 감소해 2016년부터 취약 등급으로 상향됐습니다.",
    },
    "치타": {
        "scientific_name": "Acinonyx jubatus",
        "iucn_status": "VU (취약)",
        "description": "시속 120km까지 달리는 지상 최고 속도의 동물입니다. 유전적 다양성이 매우 낮고, 서식지 파괴·인간과의 충돌로 야생 개체수가 약 7,000마리에 불과합니다.",
    },
    "재규어": {
        "scientific_name": "Panthera onca",
        "iucn_status": "VU (취약)",
        "description": "아메리카 대륙 최대의 고양이과 동물로, 장미 모양 점무늬 안에 작은 점이 있어 표범과 구별됩니다. 아마존 열대우림 파괴가 주요 위협입니다.",
    },
    # ── 신규 추가: 어류 ────────────────────────────────────────
    "황쏘가리": {
        "scientific_name": "Siniperca scherzeri",
        "iucn_status": "천연기념물 제190호 · 멸종위기 II급",
        "description": "쏘가리의 색소 변이형으로 몸 전체가 황금빛을 띠는 희귀 담수어입니다. 맑은 하천 중상류에 서식합니다.",
    },
    "열목어": {
        "scientific_name": "Brachymystax lenok",
        "iucn_status": "멸종위기 야생생물 II급",
        "description": "차고 맑은 산간 계류에 사는 연어과 냉수성 어류로, 수온 상승과 하천 오염에 매우 민감해 분포가 크게 축소되었습니다.",
    },
    "가시고기": {
        "scientific_name": "Pungitius sinensis",
        "iucn_status": "멸종위기 야생생물 II급",
        "description": "수컷이 둥지를 짓고 알과 새끼를 돌보는 부성애로 유명한 소형 담수어입니다. 등에 가시 모양 지느러미가 있습니다.",
    },
    "묵납자루": {
        "scientific_name": "Acheilognathus signifer",
        "iucn_status": "멸종위기 II급 · 한국 고유종",
        "description": "민물조개에 알을 낳아 키우는 한국 고유 납자루류로, 수질이 좋은 하천에만 서식해 환경 지표종으로 여겨집니다.",
    },
    "꾸구리": {
        "scientific_name": "Gobiobotia macrocephala",
        "iucn_status": "멸종위기 I급 · 한국 고유종",
        "description": "한강 수계에만 서식하는 한국 고유 담수어로, 머리가 크고 수염이 발달해 있습니다. 댐 건설과 하천 준설로 서식지가 크게 감소했습니다.",
    },
    "미호종개": {
        "scientific_name": "Cobitis choii",
        "iucn_status": "멸종위기 I급 · 한국 고유종",
        "description": "금강 미호천에서만 발견된 한국 고유 미꾸리과 어류입니다. 미호천 개발 및 수질 오염으로 서식 범위가 극도로 좁아졌습니다.",
    },
    "감돌고기": {
        "scientific_name": "Pseudopungtungia tenuis",
        "iucn_status": "멸종위기 I급 · 한국 고유종",
        "description": "납자루과의 한국 고유 담수어로, 꺽지 알에 기생 산란하는 독특한 번식 방법을 가집니다. 서식 하천이 매우 제한적입니다.",
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
