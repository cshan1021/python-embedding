# web
import gc
import logging
from app.core.config import settings
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
# octen
from octen import octen_ingredient
from octen import octen_products
# util
from app.utils import util_image

# 라우터 객체
api_v1_router = APIRouter()

@api_v1_router.post("/embedding_search")
async def embedding_search(
    collectionSelect: str = Form(""),
    queryText: str = Form("")    
):
    logging.info(f"collection 선택: {collectionSelect}")
    logging.info(f"조회 시작:")

    if(collectionSelect == settings.QDRANT_COLLECTION_BLOCK_INGREDIENTS):
        results = octen_ingredient.search_from_qdrant(queryText)
    else:
        results = octen_products.search_from_qdrant(queryText)

    logging.info(f"조회 종료:")

    # 메모리 정리
    gc.collect()
    
    return JSONResponse(content={"status": "success", "data": results})