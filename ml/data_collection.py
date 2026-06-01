"""
iNaturalist API로 멸종위기 포유류 이미지를 자동 수집합니다.
실행: python data_collection.py

- 이미 받은 사진은 다시 받지 않음 (관측 ID 기록)
- 나쁜 사진 지우고 재실행하면 새로운 사진으로 채워줌
"""

import os
import time
import requests
from pathlib import Path

# ── 한국 멸종위기 야생생물: 한국명 → 학명 ─────────────────────
# 동물원·수족관·생태공원·체험관 등에서 비교적 쉽게 촬영할 수 있는
# 육상동물(포유류·파충류·양서류)과 어류 중심  (조류 제외)
SPECIES = {
    # ── 포유류 (동물원·생태공원) ──────────────────────────
    "산양":           "Naemorhedus caudatus",
    "반달가슴곰":     "Ursus thibetanus",
    "사향노루":       "Moschus moschiferus",
    "수달":           "Lutra lutra",
    "여우":           "Vulpes vulpes",
    "늑대":           "Canis lupus",
    "호랑이":         "Panthera tigris",
    "표범":           "Panthera pardus",
    "스라소니":       "Lynx lynx",
    "삵":             "Prionailurus bengalensis",
    "담비":           "Martes flavigula",
    "하늘다람쥐":     "Pteromys volans",
    "대륙사슴":       "Cervus nippon",
    "붉은박쥐":       "Myotis rufoniger",
    "큰귀박쥐":       "Plecotus auritus",
    "긴꼬리딱새박쥐": "Miniopterus fuliginosus",

    # ── 파충류·양서류 (생태공원·체험관) ───────────────────
    "구렁이":         "Elaphe schrenckii",
    "남생이":         "Mauremys reevesii",
    "표범장지뱀":     "Eremias argus",
    "맹꽁이":         "Kaloula borealis",
    "금개구리":       "Pelophylax chosenicus",

    # ── 해양 포유류 (아쿠아리움·수족관) ──────────────────
    "점박이물범":     "Phoca largha",
    "상괭이":         "Neophocaena asiaeorientalis",

    # ── 국제 멸종위기종 (동물원) ──────────────────────────
    "자이언트판다":   "Ailuropoda melanoleuca",
    "눈표범":         "Panthera uncia",
    "북극곰":         "Ursus maritimus",
    "아시아코끼리":   "Elephas maximus",
    "오랑우탄":       "Pongo pygmaeus",
    "고릴라":         "Gorilla gorilla",
    "기린":           "Giraffa camelopardalis",
    "치타":           "Acinonyx jubatus",
    "재규어":         "Panthera onca",

    # ── 한국 멸종위기 어류 (수족관·아쿠아리움) ────────────
    "황쏘가리":       "Siniperca scherzeri",
    "열목어":         "Brachymystax lenok",
    "가시고기":       "Pungitius sinensis",
    "꾸구리":         "Gobiobotia macrocephala",
    "미호종개":       "Cobitis choii",
    "감돌고기":       "Pseudopungtungia tenuis",
}

MAX_IMAGES = 60    # 종당 최대 수집 이미지 수
SAVE_DIR   = "data/train"


def load_seen_ids(id_file):
    """이미 받은 관측 ID 목록 불러오기"""
    if not os.path.exists(id_file):
        return set()
    with open(id_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_seen_id(id_file, obs_id):
    """받은 관측 ID 기록"""
    with open(id_file, "a") as f:
        f.write(f"{obs_id}\n")


def download_images(korean_name, scientific_name, save_dir, max_images):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 이미 있는 사진 수 파악
    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if existing >= max_images:
        print(f"  이미 {existing}장 있음 → 건너뜀")
        return existing

    # 이미 받은 관측 ID 로드 (중복·나쁜사진 재수집 방지)
    id_file = os.path.join(save_dir, "_downloaded_ids.txt")
    seen_ids = load_seen_ids(id_file)

    downloaded = existing
    need = max_images - existing
    print(f"  기존 {existing}장 / {need}장 추가 수집 (건너뛸 ID: {len(seen_ids)}개)")
    page = 1

    while downloaded < max_images:
        try:
            resp = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params={
                    "taxon_name":    scientific_name,
                    "per_page":      30,
                    "page":          page,
                    "photos":        "true",
                    "quality_grade": "research",
                    "order_by":      "votes",
                },
                timeout=15,
            )
        except requests.RequestException as e:
            print(f"  API 요청 실패: {e}")
            break

        if resp.status_code != 200:
            print(f"  API 오류 {resp.status_code}")
            break

        results = resp.json().get("results", [])
        if not results:
            print(f"  더 이상 데이터 없음 (page {page})")
            break

        for obs in results:
            if downloaded >= max_images:
                break

            obs_id = str(obs.get("id", ""))

            # 이미 받은 적 있는 사진이면 건너뜀 (삭제한 나쁜 사진 포함)
            if obs_id in seen_ids:
                continue

            photos = obs.get("photos", [])
            if not photos:
                continue

            img_url = photos[0].get("url", "").replace("square", "medium")
            if not img_url:
                continue

            try:
                img_resp = requests.get(img_url, timeout=10)
                if img_resp.status_code == 200:
                    filepath = os.path.join(save_dir, f"{korean_name}_{downloaded:04d}.jpg")
                    with open(filepath, "wb") as f:
                        f.write(img_resp.content)
                    # ID 기록 (다음 실행 때 이 사진은 건너뜀)
                    save_seen_id(id_file, obs_id)
                    seen_ids.add(obs_id)
                    downloaded += 1
                    print(f"  [{downloaded:>3}/{max_images}] {os.path.basename(filepath)}")
            except requests.RequestException:
                pass

            time.sleep(0.3)

        page += 1
        time.sleep(1)

    return downloaded


def main():
    print("=" * 50)
    print("  멸종위기 포유류 이미지 수집 시작")
    print("=" * 50)

    total = 0
    warnings = []

    for korean_name, scientific_name in SPECIES.items():
        print(f"\n[{korean_name}]  ({scientific_name})")
        save_dir = os.path.join(SAVE_DIR, korean_name)
        count = download_images(korean_name, scientific_name, save_dir, MAX_IMAGES)
        total += count

        if count < 50:
            warnings.append(f"{korean_name}: {count}장 (부족 - 정확도 낮을 수 있음)")
        print(f"  → {count}장 완료")

    # ── 이미지가 0장인 종 폴더 정리 ──────────────────────────
    # (train.py의 ImageFolder는 빈 클래스 폴더가 있으면 오류를 내므로 제거)
    removed = []
    for name in os.listdir(SAVE_DIR):
        sdir = os.path.join(SAVE_DIR, name)
        if not os.path.isdir(sdir):
            continue
        imgs = [f for f in os.listdir(sdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not imgs:
            id_file = os.path.join(sdir, "_downloaded_ids.txt")
            if os.path.exists(id_file):
                os.remove(id_file)
            os.rmdir(sdir)
            removed.append(name)

    print("\n" + "=" * 50)
    print(f"  총 {total}장 수집 완료")
    if removed:
        print(f"\n  [제외] 이미지 0장으로 제외된 종: {', '.join(removed)}")
    if warnings:
        print("\n  [경고] 이미지 부족 종:")
        for w in warnings:
            print(f"     - {w}")
    print("=" * 50)


if __name__ == "__main__":
    main()
