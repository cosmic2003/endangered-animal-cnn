# backend/app/classifier.py

def classify_animal(image_bytes):
    """
    임시 멸종위기 동물 분류 함수 (테스트용)
    - 나중에 여기에 진짜 AI 모델이나 외부 API를 연결할 예정입니다.
    """
    print("이미지 분석 중... (임시)")
    
    # 지금은 테스트를 위해 어떤 사진을 올려도 무조건 '호랑이' 결과를 반환합니다.
    return {
        "status": "success",
        "animal_name": "호랑이 (Tiger)",
        "scientific_name": "Panthera tigris",
        "iucn_status": "EN (Endangered - 멸종위기)",
        "description": "아시아에 서식하는 대형 고양잇과 동물입니다."
    }