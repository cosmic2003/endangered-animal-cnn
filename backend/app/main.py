# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from classifier import classify_animal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "멸종위기 동물 분류 서비스 백엔드가 정상 작동 중입니다."}

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = classify_animal(image_bytes)
        if result:
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "동물 분류에 실패했습니다."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
