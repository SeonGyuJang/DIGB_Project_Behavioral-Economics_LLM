'''
[ 최종 군집별 TOPIC ]
군집 0: 교육 및 기술 관련 키워드 → 교육(education), 기술(technology), 학생(students), 언어(languages), 인공지능(AI)
군집 1: 에너지 및 자원 관리 관련 키워드 → 에너지(energy), 재료(materials), 물(water), 가스(gas), 기술(technologies)
군집 2: 역사 및 지리 연구 관련 키워드 → 역사(history), 역사가(historian), 고대(ancient), 중세(medieval), 지리(geography)
군집 3: 생물학 및 환경 연구 관련 키워드 → 동물(animals), 종(species), 환경(environmental), 과학(science), 생물학(biology)

'''
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# JSON 파일 로드
input_json_path = r"C:\Users\dsng3\Desktop\DIGB_Project\PCA\persona_clusters_pca.json"
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

def preprocess_text(text: str) -> str:
    """텍스트 전처리 함수"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip() 
    return text  

def lda_topic_modeling_for_cluster(text_list, n_topics=1, n_top_words=5, stopwords=None):
    """군집별 LDA 토픽 모델링"""
    processed = [preprocess_text(t) for t in text_list if isinstance(t, str) and t.strip()]
    
    if len(processed) == 0:  
        return [["No meaningful text"]]

    vectorizer = CountVectorizer(max_features=1000, stop_words=stopwords)
    X = vectorizer.fit_transform(processed)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics_keywords = []

    for topic_idx, topic_dist in enumerate(lda.components_):
        top_indices = topic_dist.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topics_keywords.append(top_words)

    return topics_keywords

# 사용자 지정 불용어 리스트
user_defined_stopwords = [
    "interested", "particularly", "likely", "person", "understanding", "use", "skilled", "experience", "work", "study", "individual",  # STEP1
    "potential", "knowledgeable", "able", "development", "context", "specializes", "familiar", "systems", "factors",  # STEP2
    "involved", "reclamation", "aware", "management", "health", "field",  # STEP3
    "land", "importance", "used", "impact",  # STEP4
    "background", "looking", "deep",  # STEP5
    "cancer", "methods", "security", "natural",  # STEP6
    "theorem", "space", "human",  # STEP7
    "biological", "new",  # STEP8
    "sign", "different",  # STEP9
    "language",  # STEP10
    "learning",  # STEP11
    "weapons",  # STEP12
    "wildlife",  # STEP13
    "words",  # STEP14
    "man",  # STEP15
    "keen",  # STEP16
    "production",  # STEP17
    "particular",  # STEP18
    "strong",  # STEP19
    "various",  # STEP20
    "cases",  # STEP21
    "research",  # STEP22
    "subtitling",  # STEP23
    "standards",  # STEP24
    "effectively",  # STEP25    
] 

custom_stopwords = list(set(ENGLISH_STOP_WORDS).union(user_defined_stopwords))

# 군집별 토픽 추출
all_clusters = sorted(df["cluster_label"].unique())
n_topics = 1      
n_top_words = 5  

for cluster_id in all_clusters:
    cluster_texts = df[df["cluster_label"] == cluster_id]["original_persona"].tolist()

    topic_keywords_list = lda_topic_modeling_for_cluster(
        text_list=cluster_texts,
        n_topics=n_topics,
        n_top_words=n_top_words,
        stopwords=custom_stopwords  
    )

    print(f"\n=== 군집 {cluster_id} ===")
    for t_idx, keywords in enumerate(topic_keywords_list):
        print(f" 토픽 {t_idx} 키워드: {keywords}")
