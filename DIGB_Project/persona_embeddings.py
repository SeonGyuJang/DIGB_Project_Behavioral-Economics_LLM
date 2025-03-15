import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer


# 1. 데이터 로드 (json 파일 읽기)
persona_data_path = r"C:\Users\dsng3\Desktop\DIGB_Project\persona_data.json"
with open(persona_data_path, "r", encoding="utf-8") as f:
    persona_data = json.load(f)


# 2. 전처리 함수 정의
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower() # lower_case 처리

    text = re.sub(r'[^a-z0-9\s]', '', text) # 특수문자 제거
 
    tokens = text.split() # 토큰화

    # 불용어 제거 & 표제어 추출
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)

    return " ".join(processed_tokens) # 리스트를 문자열로 변환

# 3. SentenceTransformer 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 4. 데이터 전처리 및 임베딩 계산
results = []

for entry in persona_data:
    row_idx = entry["row_idx"]  
    persona_text = entry["row"]["persona"]  

    # 텍스트 전처리
    preprocessed_text = preprocess_text(persona_text)

    # 임베딩 계산
    embedding_vector = model.encode(preprocessed_text).tolist()

    # 저장할 JSON 데이터 구조화
    results.append({
        "persona_idx": row_idx,
        "original_persona": persona_text,
        "preprocessed_persona": preprocessed_text,
        "embedding": embedding_vector
    })

# 5. JSON 파일로 저장
output_path = r"C:\Users\dsng3\Desktop\DIGB_Project\persona_embeddings.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"임베딩 페르소나 저장 완료 : {output_path}")
