import pandas as pd
import requests
import uuid
from app.core.config import settings
from collections import Counter
from qdrant_client import QdrantClient, models

# 1. 설정
excel_file_path = "./data/excel/2026.04.22_block_products.xls"
collection_name = settings.QDRANT_COLLECTION_BLOCK_PRODUCTS

# dense: Octen-Embedding-4B 모델 사용
def get_dense_embedding(text):
    response = requests.post(
        settings.OLLAMA_ENDPOINT,
        json={"model": settings.MODEL_NAME, "prompt": text}
    )
    return response.json()["embedding"]

# sparse: 자체함수
def get_sparse_embedding(text):
    text = text.lower()
    # ', ' (콤마 + 공백)를 기준으로 쪼개기
    tokens = [t.strip() for t in text.split(', ') if t.strip()]
    counter = Counter(tokens)
    
    indices = []
    values = []
    for token, count in counter.items():
        # 단어를 고유한 정수 인덱스로 변환 (해싱)
        indices.append(hash(token) % 100000) 
        values.append(float(count))
        
    return {"indices": indices, "values": values}

# Qdrant 임베딩 입력
def upsert_to_qdrant():
    try:
        client = QdrantClient(path=settings.QDRANT_PATH)

        if not client.collection_exists(collection_name):
            # Octen-Embedding-4B size=2560
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=2560, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
            )

        # 국내 반입차단 원료ㆍ성분 (엑셀 파일 읽기)
        df = pd.read_excel(excel_file_path)
        df = df.fillna("")

        points = []
        for idx, row in df.iterrows():   
            try:
                prod_name = row['제품명']
                mfg_name = row['제조사명']
                made_in = row['제조국가']
                det_ingr = row['검출성분']
                det_ingr_ko = row['검출성분(국문)']
                reg_date = row['등록일']
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{prod_name}{mfg_name}"))
                names = [name for name in [prod_name, mfg_name, det_ingr_ko] if name]
                combined_text = ", ".join(names).strip()
                point_dense_vector = get_dense_embedding(combined_text)
                point_sparse_vector = get_sparse_embedding(combined_text)

                points.append(models.PointStruct(
                    id=point_id,
                    vector= {
                        "dense": point_dense_vector,
                        "sparse": point_sparse_vector
                    },
                    payload= {
                        "prod_name": prod_name,
                        "mfg_name": mfg_name,
                        "made_in": made_in,
                        "det_ingr": det_ingr,
                        "det_ingr_ko": det_ingr_ko,
                        "reg_date": reg_date
                    }
                ))
                
                # 10개 단위로 저장 (배치 처리)
                if len(points) >= 10:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []
                    print(f"{idx + 1}개 완료...")
            except Exception as e:
                print(f"Error at {idx}: {e}")

        # 루프 종료 후 남은 포인트 저장
        if points:
            client.upsert(collection_name=collection_name, points=points)
            print(f"남은 {len(points)}개 포인트 저장 완료...")
    finally:
        client.close()

# Qdrant 임베딩 조회
def search_from_qdrant(query_text, limit=5):
    try:
        client = QdrantClient(path=settings.QDRANT_PATH)

        query_dense = get_dense_embedding(query_text)
        query_sparse = get_sparse_embedding(query_text)

        # limit 조절해서 가중치 부여
        results = client.query_points(
            collection_name = collection_name,
            prefetch = [ # Prefetch 대신 models.Prefetch 사용
                models.Prefetch(query=query_dense, using="dense", limit=40),
                models.Prefetch(query=query_sparse, using="sparse", limit=10)
            ],
            # Fusion 타입이 'RRF' 또는 'DBSF' 사용
            query=models.FusionQuery(
                fusion=models.Fusion.DBSF
            ),
            limit=limit
        )
        formatted_data = [
            {
                "id": p.id,
                "score": p.score,
                "prod_name": p.payload.get("prod_name"),
                "mfg_name": p.payload.get("mfg_name"),
                "made_in": p.payload.get("made_in"),
                "det_ingr": p.payload.get("det_ingr"),
                "det_ingr_ko": p.payload.get("det_ingr_ko"),
                "reg_date": p.payload.get("reg_date")
            }
            for p in results.points
        ]
        return formatted_data
    
    finally:
        client.close()

# embedding 실행
# python -c "from octen.octen_products import upsert_to_qdrant; upsert_to_qdrant()"