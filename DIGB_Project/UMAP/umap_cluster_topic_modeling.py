'''
[ 최종 군집별 TOPIC ]
군집 0: 생물학 및 유전 관련 키워드 → 종(species), 동물(animals), 생물학(biology), 영향(impact), 유전적(genetic)
군집 1: 에너지 및 자원 채굴 관련 키워드 → 에너지(energy), 재료(materials), 가스(gas), 시추(drilling), 생산(production)
군집 2: 건강 및 교육 관련 키워드 → 건강(health), 교육(education), 학생(students), 영양(nutrition), 모니터링(monitoring)
군집 3: 역사 및 언어 관련 키워드 → 역사(history), 역사가(historian), 중세(medieval), 언어(languages), 고대(ancient)
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
input_json_path = "/Users/jangseongyu/Desktop/DIGB_Project/persona_clusters_umap.json"
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
    "interested", "particularly", # STEP1
     "study", "understanding", "person", # STEP2
     "specializes", "knowledgeable", # STEP3
     "likely", "individual", "involved", # STEP4
     "experience", "strong", "work", # STEP5
     "familiar", "potential", "development", # STEP6
     "role", "skilled", "standards", # STEP7
     "factors", "technologies", "land", "dedicated", # STEP8
     "security", "global", # STEP9
     "looking", "natural", # STEP10
     "space", "help", # STEP11
     "reclamation", # STEP12
     "water", # STEP13
     "importance", # STEP14
     "new", "technology", # STEP15
     "aware", "environmental", # STEP16
     "use", "systems", "research", "theorem", # STEP17
     "management", "scientist", "advocate", # STEP18
     "performance", "committed", # STEP19
     "software", "deep", # STEP20
     "field", # STEP21
     "effects", # STEP22
     "state", # STEP23
     "passionate", # STEP24
     "learning", # STEP25
     "flow", # STEP26
     "school", # STEP27
     "statements", # STEP28
     "responsible", # STEP29

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
