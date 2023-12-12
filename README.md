# 올리브영 리뷰 요약 데이터를 활용한 추천시스템 성능 개선 

#### 1. 주제
리뷰 데이터를 활용하여 추천시스템 성능 개선 

#### 2. 데이터
올리브영의 리뷰 데이터

#### 3. 전처리 & EDA
사용자 피부타입 분포, 사용자 피부 고민, 사용자별 평균 평점, 카테고리 분포 

#### 4. 리뷰 데이터 처리
1. 사용자별 리뷰 concat
2. Summarization : KoBART, Lexrank, chatGPT
3. Embedding: KoBART, TF-IDF, Word2Vec => Embedding의 결과를 사용자별 Feature로 사용 

#### 5. GNN(Graph Neural Network)
- Graph 데이터 구성
- Graph Task : Link Prediction
- Graph Task : Link Regression
- Heterogeneous GNN Model

#### 6. 실험 결과
- Graph Task : Link Prediction
- Graph Task : Link Regression

#### 7. 프로젝트 의의 & 활용 방안
1. 결론
- GNN 기반 추천시스템에, 사용자의 Review 데이터를 추가하여 예측 성능을 비교
- Link Prediction에서는 성능향상이 없었으며, 기존 데이터만으로도 좋은 성능을 보임
- Link Regression에서는 Baseline 대비 낮은 RMSE를 보였으며, 상대적으로 낮은 rating을 잘 예측
- Summarize 방법은 비슷한 성능들을 보였으며, Embedding 방법은 word2vec이 준수한 성능을 보임
2. 활용방안
- 일반적으로 사용자들은 높은 rating을 주기 때문에, 낮은 rating을 받은 상품에 대한 데이터가 적음
- 사용자의 Review 데이터를 통해 성향을 파악하고, 낮은 rating에 대한 예측성능을 향상시킴으로써 잘못된 추천의 비율을 낮출수 있을 것으로 기대

