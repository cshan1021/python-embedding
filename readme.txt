# 프로젝트 생성
git init -b main
git remote add origin https://github.com/cshan1021/python-embedding
git pull origin main

# venv 환경 - Python 3.14.4
python -m venv .venv 또는 py -m venv .venv
.\.venv\Scripts\activate

# 웹서비스 - fastapi
python -m pip install fastapi uvicorn python-multipart
python -m pip install pydantic-settings
python -m pip install tldextract
python -m pip install httpx
python -m pip install jinja2

# 이미지
python -m pip install opencv-python

# qdrant db
python -m pip install qdrant-client pandas xlrd requests

# 임베딩 모델
ollama pull nicolasfer45/Octen-Embedding-4B-GGUF:latest

# 웹 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090