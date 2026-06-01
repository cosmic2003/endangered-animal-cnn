# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from classifier import classify_animal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# React 프로덕션 빌드 정적 파일 서빙
FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "build")
if os.path.exists(FRONTEND_BUILD):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD, "static")), name="static")

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

# React SPA — API 경로 외 모든 요청은 index.html로 반환
@app.get("/{full_path:path}")
async def serve_frontend(_full_path: str):
    index = os.path.join(FRONTEND_BUILD, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse(content={"message": "멸종위기 동물 분류 서비스"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
