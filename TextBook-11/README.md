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

차원축소의 필요성 : 데이터에는 중요한 부분과 중요하지 않은 부분이 존재하는데, 여기서 중요하지 않은 부분이 노이즈(noise)이다. 머신러닝 과정에서는 데이터에서 정보를 언들때 방해가 되는 불필요한 노이즈를 제거하는 것이 중요한데, 이 노이즈를 제거할 때 사용하는 방법이 차원축소(dimension reduction)이다. 차원축소는 주어진 데이터의 정보손실을 최소화하면서 노이즈를 줄이는 것이 핵심이다. 차원축소를 통해 차원이 늘어날 수록 필요한 데이터가 기하급수적으로 많아지는 차원의 저주(curse of dimensionality) 문제를 해결할 수 있다. 지도학습의 대표적인 차원축소 방법은 선형판별분석(Linear Discriminant Analysis)이 있고, 비지도학습의 대표적인 차원축소 방법은 주성분분석(Principal Component Anaysis)이 있다.<br>

# [4] 주성분 분석 (Principal Component Analysis, PCA)

PCA는 여러 특성(Feature) 변수들이 통계적으로 서로 상관관계가 없도록 변환시키는 것이다. 특성이 $p$ 개가 있을 경우, 각 특성의 벡터는 $X_1, X_2, ... X_p$ 라고 나타낼 수 있으며, 주성분분석은 오직 공분산행렬(convariance matrix) $\sum$ 에만 영향을 받는다.<br> 

# [5] 독립 성분 분석 (Independent Component Analysis, ICA)

PCA는 데이터의 분산을 최대화하는 축을 찾는 반면, ICA는 신호 간의 독립성을 기반으로 성분을 찾는다. 또한 PCA는 가우시안 분포를 가정하고 데이터의 상관관계만을 이용해 차원을 축소하거나 성분을 찾는 반면, ICA는 신호들 간의 고차원적 통계적 독립성에 초점을 맞추기 때문에 더 복잡한 구조의 신호 분리 문제를 해결할 수 있다.<br>

응용분야
 - 뇌파(EEG) 신호 분석: 뇌의 여러 부위에서 발생하는 신호들이 혼합되어 측정된 뇌파를 ICA를 사용하여 개별적인 신경 활동을 분리할 수 있다.<br>
 - 음성 신호 분리: 여러 사람이 동시에 이야기하는 상황에서, 각각의 사람의 목소리를 분리해 내는 문제를 해결하는 데 사용된다.<br>
 - 이미지 처리: 이미지에서 개별적인 패턴을 추출하거나 잡음을 제거하는 데 사용될 수 있다.<br>

알고리즘  
 - FastICA: 가장 널리 사용되는 ICA 알고리즘으로, 비선형성을 이용해 독립 성분을 빠르게 찾는 방법으로 주로 신경망 기반의 고속 수렴 특성을 이용해 구현된다. FastICA는 신호의 비정규성(정규 분포에서 얼마나 벗어나는지)을 최대화하는 방향으로 성분을 찾는데, 이는 신호가 정규 분포에서 멀어질수록 독립적인 성분일 가능성이 크다는 통계적 특성에 기반한다.<br>
 - Infomax ICA: 정보 이론을 기반으로 한 ICA 방법으로, 관측된 데이터에서 정보량을 최대화하는 방식으로 독립 성분을 추정하며, 신경망의 학습 방식과 유사한 방식으로 작동한다.<br>
 
한계
 - 혼합 행렬의 정확성: ICA는 관측 데이터가 독립적인 신호들의 선형 혼합으로 이루어졌다고 가정하지만 이 가정이 항상 맞는 것은 아니다.<br>
 - 잡음에 민감함: 데이터에 잡음이 많이 포함되어 있으면 성분 분리가 어려워질 수 있다.<br>
 - 순서 문제: ICA에서 추출된 독립 성분은 원래 신호의 순서를 보장하지 않으며, 성분의 크기도 원래 신호와 다를 수 있는데 이는 추가적인 후처리가 필요할 수 있음을 의미한다.<br>



 

# [6] 자기 조직화 지도 (Self-Organizing Maps, SOM)

# [7] 잠재 의미 분석 (Latent Semantic Analysis, LSA)

# [8] 특이값 분해 (Singular Value Decomposition, SVD)

# [9] 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA)

# [10] t-distributed Stochastic Neighbor Embedding (t-SNE)


