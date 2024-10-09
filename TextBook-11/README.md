#  11 : 비지도 학습 (Unsupervised Learning, UL) : 연관규칙, 차원축소

---

## 연관 규칙 (Association Rule)
<br>

    [1] FP Growth
    [2] 이클렛 (Eclat)
    [3] 어프라이어리 (Apriori)
  

## 차원 축소 (Dimensionality Reduction)
<br>

    [4] 주성분 분석 (Principal Component Analysis, PCA)
    [5] 독립 성분 분석 (Independent Component Analysis, ICA)
    [6] 자기 조직화 지도 (Self-Organizing Maps, SOM)
    [7] 잠재 의미 분석 (Latent Semantic Analysis, LSA)
    [8] 특이값 분해 (Singular Value Decomposition, SVD)
    [9] 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA)
    [10]t-distributed Stochastic Neighbor Embedding (t-SNE)

---  

# [1] FP Growth

# [2] 이클렛 (Eclat)

# [3] 어프라이어리 (Apriori)

---

차원축소의 필요성 : 데이터에는 중요한 부분과 중요하지 않은 부분이 존재하는데, 여기서 중요하지 않은 부분이 노이즈(noise)이다. 머신러닝 과정에서는 데이터에서 정보를 언들때 방해가 되는 불필요한 노이즈를 제거하는 것이 중요한데, 이 노이즈를 제거할 때 사용하는 방법이 차원축소(dimension reduction)이다. 차원축소는 주어진 데이터의 정보손실을 최소화하면서 노이즈를 줄이는 것이 핵심이다. 차원축소를 통해 차원이 늘어날 수록 필요한 데이터가 기하급수적으로 많아지는 차원의 저주(curse of dimensionality) 문제를 해결할 수 있다. 지도학습의 대표적인 차원축소 방법은 선형판별분석(Linear Discriminant Analysis)이 있고, 비지도학습의 대표적인 차원축소 방법은 주성분분석(Principal Component Anaysis)이 있다.

# [4] 주성분 분석 (Principal Component Analysis, PCA)

주성분분석은 여러 특성(Feature) 변수들이 통계적으로 서로 상관관계가 없도록 변환시키는 것이다. 특성이 $p$ 개가 있을 경우, 각 특성의 벡터는 $X_1, X_2, ... X_p$ 라고 나타낼 수 있으며, 주성분분석은 오직 공분산행렬(convariance matrix) $\sum$ 에만 영향을 받는다. 

# [5] 독립 성분 분석 (Independent Component Analysis, ICA)

# [6] 자기 조직화 지도 (Self-Organizing Maps, SOM)

# [7] 잠재 의미 분석 (Latent Semantic Analysis, LSA)

# [8] 특이값 분해 (Singular Value Decomposition, SVD)

# [9] 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA)

# [10] t-distributed Stochastic Neighbor Embedding (t-SNE)


