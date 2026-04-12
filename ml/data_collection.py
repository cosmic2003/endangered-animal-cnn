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

# ── 한국 멸종위기 포유류 14종: 한국명 → 학명 ──────────────────
SPECIES = {
    "산양":           "Naemorhedus caudatus",
    "반달가슴곰":     "Ursus thibetanus",
    "사향노루":       "Moschus moschiferus",
    "붉은박쥐":       "Myotis rufoniger",
    "수달":           "Lutra lutra",
    "여우":           "Vulpes vulpes",
    "늑대":           "Canis lupus",
    "호랑이":         "Panthera tigris",
    "표범":           "Panthera pardus",
    "삵":             "Prionailurus bengalensis",
    "담비":           "Martes flavigula",
    "하늘다람쥐":     "Pteromys volans",
    "큰귀박쥐":       "Plecotus auritus",
    "긴꼬리딱새박쥐": "Miniopterus fuliginosus",
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
            warnings.append(f"{korean_name}: {count}장 (부족 — 정확도 낮을 수 있음)")
        print(f"  → {count}장 완료")

    print("\n" + "=" * 50)
    print(f"  총 {total}장 수집 완료")
    if warnings:
        print("\n  ⚠️  이미지 부족 종:")
        for w in warnings:
            print(f"     - {w}")
    print("=" * 50)


if __name__ == "__main__":
    main()
