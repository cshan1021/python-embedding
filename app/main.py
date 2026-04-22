# web
import logging
import uvicorn
import sys
from app.api.v1.api_v1_router import api_v1_router
from app.core.config import settings
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(settings.PROJECT_NAME)

# app
app = FastAPI()
app.mount("/static", StaticFiles(directory=settings.STATIC_PATH), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_PATH)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# router
app.include_router(api_v1_router, prefix="/api/v1", tags=["api-v1"])

if __name__ == "__main__":
    # 8090 포트로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8090)

# python -m app.main