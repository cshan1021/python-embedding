import pandas as pd
import requests
import uuid
from app.core.config import settings
from collections import Counter
from qdrant_client import QdrantClient, models

# 1. 설정
excel_file_path = "./data/excel/2026.04.22_block_ingredients.xls"
client = QdrantClient(path=settings.QDRANT_PATH)
collection_name = settings.QDRANT_COLLECTION_BLOCK_INGREDIENTS

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
                status_name = row['구분']
                ko_name = row['원료ㆍ성분명(한글)']
                en_name = row['원료ㆍ성분명(영문)']
                etc_name = row['기타명칭']
                status_date = row['지정ㆍ해제일']
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ko_name}{en_name}"))
                names = [name for name in [ko_name, en_name, etc_name] if name]
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
                        "status_name": status_name,
                        "ko_name": ko_name,
                        "en_name": en_name,
                        "etc_name": etc_name,
                        "status_date": status_date
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
        return results.points
    
    finally:
        client.close()

# embedding
upsert_to_qdrant()


query_text = """
Serving Size: 2 Capsules Servings Per Container: 30

Amount per Serving	% Daily Value
Calcium (as Calcium Carbonate)	155 mg
Proprietary Blend	677 mg
L-Arginine HCl	
Gymnema Sylvestre
Damiana (Turnera diffusa) (Leaf)	
Yohimbe (Pausinystalia yohimbe) (Bark)	
Maca (Root)	
Other Ingredients: Gelatin (Capsule) and Magnesium Stearate.

Mfg for Mellora, Inc. 818 SW Third Ave, Suite 81 Portland, OR 97204 www.lyriana.com

results = search_from_qdrant(query_text)
for idx, result in enumerate(results):
    payload = result.payload
    print(f"[{idx+1}위] 점수: {result.score:.4f}")
    print(f"성분명: {payload.get('ko_name')} ({payload.get('en_name')})")
    print(f"상태: {payload.get('status_name')} | 날짜: {payload.get('status_date')}")
    print(f"기타명칭: {payload.get('etc_name')}")
    print("-" * 50)
"""

# 개별실행
# python -m octen.octen_ingredient