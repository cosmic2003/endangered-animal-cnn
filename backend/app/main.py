# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms  # 수정: torchvision.transforms로 정확히 임포트
import torch
from PIL import Image
import io

# CORS 에러를 해결하기 위한 패키지 임포트
from fastapi.middleware.cors import CORSMiddleware

# 주의: classifier.py 파일이 같은 폴더 내에 존재해야 합니다.
from classifier import classify_animal 

app = FastAPI()

# ==========================================
# 🚨 CORS (교차 출처 리소스 공유) 허용 설정
# 리액트(3000번 포트)에서 오는 요청을 안전하다고 판단하여 허락해 줍니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST 등 모든 요청 방식 허용
    allow_headers=["*"],  # 모든 헤더 허용
)
# ==========================================

# 사전 학습된 ResNet50 모델 불러오기 (임시 테스트용)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()  # 평가 모드로 설정

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    """
    메인 엔드포인트.
    """
    return {"message": "멸종위기 동물 분류 서비스 백엔드가 정상 작동 중입니다."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    업로드된 이미지에 대해 기본 ResNet50 예측을 수행합니다.
    """
    try:
        # 업로드된 파일을 비동기로 읽고 메모리에서 PIL 이미지로 변환
        image_bytes = await file.read()
        # .convert("RGB")를 추가하여 png(투명도 포함) 파일 업로드 시 발생하는 에러 방지
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 이미지 전처리
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

        # 모델 추론
        with torch.no_grad():
            output = model(input_batch)

        # 결과를 반환할 형식으로 변환
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 예측 클래스 인덱스 가져오기
        predicted_class = torch.argmax(probabilities).item()

        # 예측 결과 반환
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist()
        })
        
    except Exception as e:
        return JSONResponse(content={"error": f"이미지 처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"}, status_code=500)

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """
    업로드된 이미지에서 멸종위기 동물을 분류하고 IUCN Red List 정보를 반환합니다.
    """
    try:
        # FastAPI의 UploadFile 객체에서 바이트 데이터를 추출
        image_bytes = await file.read()
        
        # classifier.py의 함수에 바이트 데이터 전달
        result = classify_animal(image_bytes)
        
        if result:
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "동물 분류 및 IUCN 정보 조회에 실패했습니다."}, status_code=400)
            
    except Exception as e:
        return JSONResponse(content={"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}, status_code=500)

# 모든 라우터(엔드포인트) 정의가 끝난 후 파일의 가장 마지막에 실행 블록을 배치합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)