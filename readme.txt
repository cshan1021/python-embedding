# 프로젝트 생성
git init -b main
git remote add origin https://github.com/cshan1021/python-embedding
git pull origin main

# venv 환경 - Python 3.14.4
python -m venv .venv 또는 py -m venv .venv
.\.venv\Scripts\activate

python -m pip install qdrant-client pandas xlrd requests

ollama pull nicolasfer45/Octen-Embedding-4B-GGUF:latest
