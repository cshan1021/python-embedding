# web
import gc
import logging
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
# octen

# util
from app.utils import util_image

# 라우터 객체
api_v1_router = APIRouter()

@api_v1_router.post("/analyze_images")
async def analyze_images(
    servingSelect: str = Form(""),
    modelSelect: str = Form(""),
    files: List[UploadFile] = File(...)
):
    logging.info(f"서빙 선택: {servingSelect}")
    logging.info(f"모델 선택: {modelSelect}")
 
    logging.info(f"분석 시작:")

    logging.info(f"분석 종료:")

    results = {}

    # 메모리 정리
    gc.collect()
    
    return JSONResponse(content={"status": "success", "data": results})