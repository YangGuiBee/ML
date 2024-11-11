#  11 : 비지도 학습(Unsupervised Learning, UL) : 연관규칙, 차원축소

---

## 연관 규칙(Association Rule)
<br>

    [1] Apriori
    [2] FP-Growth(Frequent Pattern Growth) 
    [3] Eclat(Equivalence Class Transformation)
    [4] Multi-level Association Rules
    [5] Multi-dimensional Association Rules
    [6] AIS(Artificial Immune System)
    [7] SETM(Sequential Execution of Transaction Merging)

    

## 차원 축소(Dimensionality Reduction)
<br>

    [1] PCA(Principal Component Analysis)
    [2] t-SNE(t-distributed Stochastic Neighbor Embedding)
    [3] UMAP(Uniform Manifold Approximation and Projection)
    [4] SVD(Singular Value Decomposition)
    [5] ICA(Independent Component Analysis)
    [6] LDA(Linear Discriminant Analysis)
    [7] Isomap
    [8] MDS(Multidimensional Scaling)
    [9] LSA(Latent Semantic Analysis)
    [10] SOM(Self-Organizing Maps)

    [차원 축소 알고리즘 평가방법]
    ▣ 재구성 오류(Reconstruction Error) : 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)
    ▣ 분산 유지율(Explained Variance Ratio) : 각 주성분이 설명하는 분산 비율로 데이터의 정보 손실정도 파악
    ▣ 상호 정보량(Mutual Information) :  차원 축소 전후 데이터의 정보량을 비교
    ▣ 군집 평가 지표 : Silhouette Score, Davies-Bouldin Index, 실제 레이블과 예측 레이블 비교(ARI, NMI)

---  

# [1] Apriori
▣ 정의 : 연관규칙 학습을 위한 고전적인 알고리즘으로, 빈발항목 집합(frequent itemsets)을 찾아내고 그 집합들 간의 연관성을 추출<br>
▣ 필요성 : 대규모 데이터에서 연관성을 발견하는 작업은 계산 비용이 높을 수 있는데, Apriori는 빈발하지 않은 항목 집합을 먼저 제거해 검색 공간을 줄여주는 방식으로 효율적인 탐색이 가능<br>
▣ 장점 : 간단한 구조로 이해하기 쉽고, 계산 공간을 줄이기 위한 사전 단계를 가지고 있어, 효율적인 탐색이 가능<br>
▣ 단점 : 탐색 공간이 커지면 성능이 저하되고 대규모 데이터에서 비효율적일 수 있으며, 매번 새로운 후보집합을 생성해야 하므로 계산비용이 크다.<br>
▣ 응용분야 : 시장 바구니 분석(장바구니 데이터에서 자주 함께 구매되는 상품을 찾음), 추천 시스템, 웹 페이지 연결성 분석<br>
▣ 모델식 : 지지도(Support): 특정 항목 집합이 전체 거래에서 발생하는 빈도, 신뢰도(Confidence): 특정 항목이 발생한 경우 다른 항목이 함께 발생할 확률, 향상도(Lift): 항목 간의 상호의존성을 측정<br>

    from mlxtend.frequent_patterns import apriori, association_rules
    import pandas as pd

    # 예시 데이터 생성 (장바구니 데이터)
    data = {'milk': [1, 0, 1, 1, 0],
            'bread': [1, 1, 1, 0, 1],
            'butter': [0, 1, 0, 1, 0], 
            'beer': [1, 1, 1, 0, 0]}
    df = pd.DataFrame(data)

    # Apriori 알고리즘을 사용하여 빈발 항목 집합 찾기
    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

    # 연관 규칙 찾기
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    print(frequent_itemsets)
    print(rules)

<br>

# [2] FP-Growth(Frequent Pattern Growth)
▣ 정의: Apriori 알고리즘의 대안으로 FP-Tree(Frequent Pattern Tree)를 통해 빈발항목 집합을 생성하는 알고리즘으로, Apriori와 달리 매번 후보집합을 생성하지 않으며, 데이터의 트랜잭션을 직접 탐색하여 빈발항목 집합을 구한다.<br>
▣ 필요성: Apriori의 성능 문제를 해결하기 위해 고안<br>
▣ 장점: 메모리 효율이 높고, 대규모 데이터셋에서 빠르게 작동<br>
▣ 단점: FP-트리 구조를 구축하는 데 추가 메모리가 필요하며, 구현이 복잡하고 FP-Tree 생성을 위한 학습이 필요<br>
▣ 응용분야: 대규모 데이터 분석, 전자상거래 추천 시스템<br>

    from mlxtend.frequent_patterns import fpgrowth

    # FP-Growth 알고리즘을 사용하여 빈발 항목 집합 찾기
    frequent_itemsets_fp = fpgrowth(df, min_support=0.6, use_colnames=True)

    # 연관 규칙 찾기
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)

    print(frequent_itemsets_fp)
    print(rules_fp)
    
<br>

# [3] Eclat(Equivalence Class Transformation)
▣ 정의: Apriori와 FP-Growth의 대안으로, 트랜잭션 간의 공통항목(교집합)을 기반으로 빈발항목을 추출하는 알고리즘<br>
▣ 필요성: 데이터의 수가 많아도 트랜잭션 간 교차 계산을 통해 효율적으로 연관 규칙을 도출<br>
▣ 장점 : 수평적 데이터 구조를 이용하여 트랜잭션 데이터에서 빈발 항목 집합을 빠르게 찾고, 저장 공간을 효율적으로 사용하며, 교차 연산을 통해 빈발 항목을 추출<br>
▣ 단점 : 트랜잭션 ID 집합을 계속 업데이트해야 하므로 메모리 사용이 증가할 수 있으며, 대규모 데이터셋에서는 효율성이 떨어질 수 있음<br>
▣ 응용분야 : 대규모 데이터에서 빈발 패턴 분석, 웹 클릭 로그 분석, 텍스트 마이닝에서 자주 나타나는 단어 조합 분석<br>
▣ 모델식 : 항목 집합의 지지도 계산을 위해 트랜잭션 ID 집합의 교집합을 사용하며 빈발항목 집합의 지지도를 계산할 때 교집합을 통해 빈발 항목을 찾아낸다<br>

    from itertools import combinations

    # 데이터 집합에서 항목별로 트랜잭션 ID 집합 생성
    def tid_lists(df):
        tid_dict = {}
        for col in df.columns:
            tid_dict[col] = set(df.index[df[col] == 1])
        return tid_dict

    # 빈발항목 집합을 찾는 Eclat 알고리즘 구현
    def eclat(tid_dict, min_support=0.6):
        n_transactions = len(df)
        frequent_itemsets = {}
    
        for k in range(1, len(tid_dict)+1):
            for comb in combinations(tid_dict.keys(), k):
                intersect_tids = set.intersection(*[tid_dict[item] for item in comb])
                support = len(intersect_tids) / n_transactions
                if support >= min_support:
                    frequent_itemsets[comb] = support
        return frequent_itemsets

    # 트랜잭션 ID 집합 계산
    tid_dict = tid_lists(df)

    # Eclat 알고리즘 실행
    frequent_itemsets_eclat = eclat(tid_dict, min_support=0.6)
    print(frequent_itemsets_eclat)
    
<br>

# [4] Multi-level Association Rules
▣ 정의: 연관 규칙을 계층적으로 탐색하여 다중 수준에서 규칙을 생성하는 방식<br>
▣ 필요성: 제품 카테고리별 분석이 필요한 경우에 적합<br>
▣ 장점: 더 정교한 규칙을 생성<br>
▣ 단점: 복잡성이 증가하며, 해석이 어려워질 수 있음<br>
▣ 응용분야: 전자상거래, 추천 시스템, 마케팅 분석<br>

<br>

# [5] Multi-dimensional Association Rules
▣ 정의: 여러 속성을 포함하여 다양한 차원의 규칙을 생성<br>
▣ 필요성: 다양한 속성 간 관계를 탐색하는 데 적합<br>
▣ 장점: 규칙의 범위를 확장할 수 있어 더 세밀한 규칙 도출 가능.<br>
▣ 단점: 복잡성과 해석의 어려움<br>
▣ 응용분야: 사용자 속성 기반 추천 시스템, 마케팅 인텔리전스<br>

<br>

# [6] SETM(Sequential Execution of Transaction Merging)
▣ 정의: Apriori의 변형으로, 데이터베이스 접근 횟수를 줄여 성능을 개선<br>
▣ 필요성: 연관 규칙 생성에서 데이터베이스 접근이 빈번한 경우 사용<br>
▣ 장점: Apriori에 비해 빠르며 메모리 효율적<br>
▣ 단점: 성능 한계로 인해 많이 사용되진 않음<br>
▣ 응용분야: 실시간 데이터 분석, 연관 규칙 학습<br>

<br>

# [7] AIS(Artificial Immune System)
▣ 정의: 거래 데이터를 순차적으로 결합하여 빈번한 항목 집합을 찾는 초기 연관규칙 알고리즘 중 하나<br>
▣ 필요성: 초기 연관 규칙 연구에서 활용되었으나, 성능의 한계로 현재는 거의 사용되지 않음<br>
▣ 장점: 간단한 구조로 이해하기 쉽다.<br>
▣ 단점: 비효율적이며, Apriori보다 성능이 떨어짐<br>
▣ 응용분야: 초기 연관 규칙 연구<br>

<br>


---

**차원축소의 필요성 :** 데이터에 포함된 노이즈(noise)를 제거할 때 사용하는 방법이 차원축소(dimension reduction)이다. 차원축소는 주어진 데이터의 정보손실을 최소화하면서 노이즈를 줄이는 것이 핵심이다. 차원축소를 통해 차원이 늘어날 수록 필요한 데이터가 기하급수적으로 많아지는 차원의 저주(curse of dimensionality) 문제를 해결할 수 있다. 지도학습의 대표적인 차원축소 방법은 선형판별분석(Linear Discriminant Analysis)이 있고, 비지도학습의 대표적인 차원축소 방법은 주성분분석(Principal Component Anaysis)이 있다.<br>

# [1] PCA(Principal Component Analysis)
▣ 정의 : 데이터의 분산을 최대한 보존하면서 데이터의 주요 성분(주성분)을 찾기 위해 선형 변환을 적용하는 차원 축소 알고리즘. 여러 특성(Feature) 변수들이 통계적으로 서로 상관관계가 없도록 변환시키는 것으로 고차원 데이터를 저차원으로 변환하는 차원 축소 기법. 주성분분석은 오직 공분산행렬(convariance matrix) $\sum$ 에만 영향을 받는다.<br> 
▣ 장점 : 정보 손실을 최소화하면서 고차원 데이터를 저차원으로 축소, 데이터의 잡음을 효과적으로 제거, 고차원 데이터를 저차원으로 변환하여 데이터의 구조를 쉽게 이해하고 분석<br>
▣ 단점 : 선형 변환만을 가정(커널PCA 같은 비선형 변형 기법이 필요), 각 주성분이 원래 데이터의 어떤 특성을 설명하는지 직관적으로 해석하기 어렵다. 분산에 중요한 정보가 있을 경우 이를 놓칠 수 있다.<br>
▣ 응용분야 : 고차원 데이터를 2D 또는 3D로 변환해 데이터의 패턴을 직관적으로 시각화, 잡음 제거, 얼굴 인식에서 얼굴 이미지의 주요 특징을 추출하여 얼굴을 효율적으로 분류<br>
▣ 모델식 : 주성분은 공분산 행렬의 고유값과 고유벡터를 사용하여 계산<br>
데이터 행렬 𝑋의 공분산 행렬 𝐶의 고유값과 고유벡터를 통해 새로운 주성분을 계산 : $C=\frac{1}{n-1}X^TX$<br>
고유값 분해(v_i는 i번째 고유벡터, \lambda_i는 i번째 고유값) : $Cv_i = \lambda_iv_i$<br>
▣ PCA의 절차 : 분산의 최대화: 주성분은 데이터의 분산(변동성)을 최대한 많이 설명할 수 있는 방향으로 정해진다. 데이터의 주요한 변동성을 나타내는 축을 먼저 찾고, 그 축을 기준으로 데이터를 투영한다. 직교성: 각 주성분은 서로 직교(orthogonal)해야 하는데 이는 각 주성분이 서로 상관관계가 없는 독립적인 축이라는 것을 의미한다.<br>
(1) 데이터 표준화 : PCA를 수행하기 전에 데이터의 스케일을 맞추기 위해 각 변수의 평균을 0으로 만들고 분산을 1로 맞추는 z-점수 정규화 과정<br>
(2) 공분산 행렬계산 : 공분산(두 변수가 함께 변하는 정도) 행렬 계산을 통해 데이터의 분산이 어떻게 다른 변수들과 상호작용하는지 확인<br>
  $Cov(X,Y)=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})(Y_i-\overline{Y})$<br>
(3) 고유값 분해(Eigenvalue Decomposition) : 공분산 행렬의 고유벡터(eigenvector)는 PCA의 주성분에 해당, 고유값(eigenvalue)은 주성분이 설명하는 분산의 양을 나타냄<br>
(4) 주성분 선택: 고유값이 큰 순서대로 주성분을 선택(가장 큰 고유값에 해당하는 고유벡터가 제1주성분, 그다음 고유값이 제2주성분 : 고유값이 큰 주성분일수록 데이터의 분산설명렬이 높다)<br>
(5) 차원 축소: 선택된 주성분을 사용해 데이터를 저차원으로 투영. 데이터의 중요한 특성(분산)을 유지하면서 불필요한 차원을 제거하여 차원을 축소<br>
 
<br>

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # PCA 적용
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA on Iris Dataset")
    plt.colorbar()
    plt.show()

    # 분산 유지율 출력
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("Total Variance Retained:", sum(pca.explained_variance_ratio_))

<br>

# [2] t-SNE(t-distributed Stochastic Neighbor Embedding)
▣ 정의: 고차원 데이터의 국소 구조를 잘 보존하여 저차원으로 투영하는 비선형 차원 축소 알고리즘<br>
▣ 필요성: 데이터의 클러스터 구조를 유지한 채 저차원으로 투영하여 데이터 간의 관계를 시각적으로 파악하기 위해 사용<br>
▣ 장점 : 고차원 데이터의 군집 구조를 잘 반영하여 데이터의 숨겨진 패턴을 시각적으로 잘 드러내고, 비선형 구조를 가진 데이터에서도 효과적으로 작동<br>
▣ 단점 : 데이터 포인트 수가 많아질수록 계산 시간이 급격히 증가하고, 초기 매개변수(예: σ 값 및 학습률)에 민감하게 반응<br> 
▣ 응용 분야 : 이미지 데이터, 텍스트 데이터, 유전자 표현 데이터 등의 시각화, 클러스터링 분석, 데이터 전처리, 신경망 모델의 중간 출력을 시각화<br>
▣ 모델식: 고차원 데이터의 유사도와 저차원 데이터의 유사도 분포를 맞추기 위해 코스트 함수 𝐾𝐿(𝑝∥𝑞)를 최소화<br>

<br>

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data.target)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE on Iris Dataset")
    plt.colorbar()
    plt.show()

<br>

# [5] 독립 성분 분석(Independent Component Analysis, ICA)
▣ 정의 : ICA는 다변량 신호에서 통계적으로 독립적인 성분을 추출하는 비선형 차원 축소 기법으로 주로 관찰된 신호를 독립적인 원천 신호로 분리하는 데 사용된다. PCA는 데이터의 분산을 최대화하는 축을 찾는 반면, ICA는 신호 간의 독립성을 기반으로 성분을 찾는다. 또한 PCA는 가우시안 분포를 가정하고 데이터의 상관관계만을 이용해 차원을 축소하거나 성분을 찾는 반면, ICA는 신호들 간의 고차원적 통계적 독립성에 초점을 맞추기 때문에 더 복잡한 구조의 신호 분리 문제를 해결할 수 있다.<br>
▣ ICA의 필요성 : 관측된 신호가 여러 독립적인 원천 신호의 혼합으로 구성될 때, 각 독립적인 신호를 복원하는 데 필요하며 특히 신호 처리 및 음성 분리에 유용하다.<br>
▣ ICA의 응용분야
 - 뇌파(EEG) 신호 분석: 뇌의 여러 부위에서 발생하는 신호들이 혼합되어 측정된 뇌파를 ICA를 사용하여 개별적인 신경 활동을 분리할 수 있다.<br>
 - 음성 신호 분리: 여러 사람이 동시에 이야기하는 상황에서, 각각의 사람의 목소리를 분리해 내는 문제를 해결하는 데 사용된다.<br>
 - 이미지 처리: 이미지에서 개별적인 패턴을 추출하거나 잡음을 제거하는 데 사용될 수 있다.<br>
▣ ICA의 ICA의 알고리즘  
 - FastICA: 가장 널리 사용되는 ICA 알고리즘으로, 비선형성을 이용해 독립 성분을 빠르게 찾는 방법으로 주로 신경망 기반의 고속 수렴 특성을 이용해 구현된다. FastICA는 신호의 비정규성(정규 분포에서 얼마나 벗어나는지)을 최대화하는 방향으로 성분을 찾는데, 이는 신호가 정규 분포에서 멀어질수록 독립적인 성분일 가능성이 크다는 통계적 특성에 기반한다.<br>
 - Infomax ICA: 정보 이론을 기반으로 한 ICA 방법으로, 관측된 데이터에서 정보량을 최대화하는 방식으로 독립 성분을 추정하며, 신경망의 학습 방식과 유사한 방식으로 작동한다.<br>
▣ 모델식 : ICA는 신호의 혼합 행렬 𝑋를 독립적인 신호 𝑆와 혼합 행렬 𝐴로 분해하는 문제로 정의하며, 목표는 𝑆를 추출<br> 
▣ ICA의 장점 : 통계적으로 독립적인 신호를 분리할 수 있으며 신호 처리, 이미지 분할, 음성 분리 등에서 강력한 성능을 발휘한다. 
▣ ICA의 ICA의 단점
 - 혼합 행렬의 정확성: ICA는 관측 데이터가 독립적인 신호들의 선형 혼합으로 이루어졌다고 가정하지만 이 가정이 항상 맞는 것은 아니다.<br>
 - 잡음에 민감함: 데이터에 잡음이 많이 포함되어 있으면 성분 분리가 어려워질 수 있다.<br>
 - 순서 문제: ICA에서 추출된 독립 성분은 원래 신호의 순서를 보장하지 않으며, 성분의 크기도 원래 신호와 다를 수 있는데 이는 추가적인 후처리가 필요할 수 있음을 의미한다.<br>
▣ ICA의 모델식 : ICA는 신호의 혼합 행렬 𝑋를 독립적인 신호 𝑆와 혼합 행렬 𝐴로 분해하는 문제로 정의 : $𝑋=𝐴𝑆$<br>


<br>
    from sklearn.decomposition import FastICA
    import numpy as np

    # 예시 데이터 생성
    data = np.random.rand(100, 5)

    # ICA 적용
    ica = FastICA(n_components=2)
    ica_result = ica.fit_transform(data)

    # 결과 출력
    print(ica_result)

<br> 

# [6] 자기 조직화 지도(Self-Organizing Maps, SOM)
▣ SOM의 정의 : SOM은 비지도 학습 알고리즘 중 하나로, 고차원의 데이터를 저차원(일반적으로 2차원) 공간으로 투영하여 데이터의 구조를 시각화하는 데 사용된다. PCA는 선형 변환을 통해 차원 축소를 수행하지만, SOM은 비선형 변환을 사용하여 더 복잡한 데이터 구조를 반영할 수 있으며, k-평균은 각 군집의 중심을 찾는 방식으로 군집화를 수행하는 반면, SOM은 뉴런이 격자 형태로 조직되어 있어 더 직관적인 시각화가 가능하다.<br> 
▣ SOM의 핵신개념
 - 차원 축소: 고차원의 데이터를 2차원 맵으로 변환하여 시각화하여 고차원 데이터의 구조를 쉽게 이해하고 분석하는 데 유용<br> 
 - 군집화: SOM은 데이터를 자연스럽게 군집화하는 효과를 가지고 있으며, 유사한 패턴을 가진 데이터는 SOM의 인접한 뉴런들에 할당<br> 
 - 비선형 변환: SOM은 데이터의 비선형 구조도 반영하여 선형적인 변환 기법보다 복잡한 데이터 구조를 효과적으로 다룰 수 있는 특징<br>
▣ SOM의 모델식 : SOM은 뉴런의 위치 𝑟와 입력 벡터 𝑥 간의 거리 함수로 클러스터를 형성<br>
학습 과정에서 뉴런의 가중치 𝑤_𝑖 : $w_i(t+1)=w_i(t)+η(t)h(t)(x(t)−w_i(t))$, 𝜂(𝑡)는 학습률, ℎ(𝑡)는 이웃 함수<br>
▣ SOM의 절차
 - (1) 초기화: SOM의 각 뉴런에 임의의 가중치 벡터를 할당(이 가중치 벡터는 입력 데이터와 같은 차원)<br>
 - (2) 입력 데이터 선택: 학습 과정에서 입력 데이터 벡터 하나를 무작위로 선택<br>
 - (3) 승자 뉴런(BMU, Best Matching Unit) 찾기: SOM의 모든 뉴런 중에서 현재 입력 벡터와 가장 유사한(가중치 벡터 간의 유클리드 거리로 측정) 뉴런을 찾는 경쟁 학습의 핵심 단계<br>
 - (4) 가중치 벡터 갱신: 선택된 승자 뉴런과 그 주변 이웃 뉴런들의 가중치 벡터를 조정한다. 이때, 가중치 벡터는 입력 데이터에 더 가깝게 이동<br> 
▣ SOM의 장점
 - 비지도 학습: SOM은 라벨링되지 않은 데이터를 학습할 수 있으므로, 데이터에 대한 사전 정보가 없어도 유용하게 사용 가능<br> 
 - 시각화 및 직관적 분석: 고차원 데이터를 저차원 맵으로 변환하여 데이터를 시각적으로 분석할 수 있으며, 군집의 분포나 데이터의 경향성을 직관적으로 이해<br> 
 - 이웃 구조 보존: SOM은 입력 데이터의 이웃 관계를 보존하면서 저차원으로 투영하므로, 원래 데이터의 공간적 관계를 유지<br> 
▣ SOM의 단점
 - 복잡한 파라미터 설정: 학습률, 이웃 크기 등 여러 파라미터를 적절히 설정해야 하며, 잘못된 설정은 모델의 성능에 부정적인 영향<br> 
 - 대규모 데이터 학습에 비효율적: SOM은 입력 데이터가 많거나 차원이 매우 높을 때 학습 시간이 길어질 가능성<br> 
 - 해석의 어려움: SOM의 결과는 비선형 변환을 통해 얻어진 것이므로, 변환된 맵을 해석하는 것이 PCA 등의 선형 변환보다 더 어려움<br> 
▣ SOM의 응용분야
 - 이미지 분석: 이미지에서 패턴을 추출하거나, 유사한 이미지들을 군집화하는 데 사용된다.<br>
 - 문서 분류: 텍스트 데이터를 저차원 공간에 투영하여 유사한 문서를 군집화할 수 있다.<br>
 - 음성 인식: 음성 데이터를 학습하여 유사한 음성 패턴을 인식하고 분류하는 데 사용된다.<br>
 - 생물정보학: 유전자 데이터를 시각화하고 패턴을 찾아내는 데 활용된다.<br>
▣ SOM의 모델식 : SOM은 뉴런의 위치 𝑟와 입력 벡터 𝑥 간의 거리 함수로 클러스터를 형성(𝜂(𝑡)는 학습률, ℎ(𝑡)는 이웃 함수)<br>
$W(t+1)=W(t)+\theta(t)\cdot\eta(t)\cdot(X-W(t))$<br>


    from minisom import MiniSom
    import numpy as np

    #예시 데이터 생성
    data = np.random.rand(100, 5)

    #SOM 정의 및 학습
    som = MiniSom(10, 10, 5, sigma=0.3, learning_rate=0.5)
    som.train_random(data, 100)

    #SOM 시각화
    from pylab import plot, show, colorbar, bone
    bone()
    for i, x in enumerate(data):
        w = som.winner(x)
        plot(w[0]+0.5, w[1]+0.5, 'ro')
    show()

<br> 

# [7] 잠재 의미 분석(Latent Semantic Analysis, LSA)
▣ LSA의 정의 : LSA는 텍스트 데이터를 분석할 때 사용되는 자연어 처리(Natural Language Processing, NLP) 기법으로, 문서 간의 잠재적인 의미적 관계를 분석하고, 차원을 축소하여 텍스트의 중요한 의미적 패턴을 파악하는 방법이다. LSA는 특히 문서내 단어들 간의 공통적인 의미 구조를 찾는 데 중점을 두며, 이를 통해 문서 간의 유사성을 계산하고, 텍스트 데이터에서 의미적 관계를 도출한다.<br>
▣ LSA의 핵심개념
 - 의미 공간: LSA는 문서와 단어 간의 관계를 다차원 공간에서 표현한다. 여기서 각 단어와 문서는 고차원 벡터로 표현되며, LSA는 이러한 벡터들을 더 작은 의미 공간으로 투영해 문서와 단어 간의 잠재적인 의미적 관계를 파악한다.<br>
 - 차원 축소: LSA의 핵심 아이디어는 고차원의 단어-문서 행렬을 더 작은 차원으로 축소한다. 이렇게 하면 텍스트 데이터의 주요 패턴이 유지되면서도 불필요한 노이즈와 차원이 제거된다. 이를 위해 LSA는 주로 특이값 분해(Singular Value Decomposition, SVD)를 사용한다.<br>
 - 잠재 의미: LSA는 문서 내 단어들의 공통적인 의미 패턴을 학습하여, 단어가 명시적으로 나오지 않더라도 그 단어의 잠재적 의미를 파악할 수 있다. 예를 들어, "컴퓨터"와 "노트북"이라는 단어가 직접적인 상관관계를 가지지 않더라도 LSA는 이들이 비슷한 맥락에서 사용될 가능성을 학습한다.<br>
▣ LSA의 장점
 - 차원 축소: LSA는 텍스트 데이터의 고차원성을 효과적으로 축소하면서도 중요한 의미적 관계를 유지한다.<br>
 - 노이즈 제거: 텍스트 데이터에서 불필요한 노이즈를 제거하고, 중요한 의미 패턴을 추출하는 데 유리하다.<br>
 - 숨겨진 의미 관계 파악: 직접적으로 나타나지 않은 단어들 간의 의미적 유사성을 찾아내는 데 효과적이다.<br>
 - 문서 간 유사성 계산: LSA는 문서 간의 의미적 유사성을 쉽게 계산할 수 있으며, 이는 정보 검색 및 문서 추천 시스템에서 유용하게 사용된다.<br>
▣ LSA의 단점
 - 단어 순서 무시: LSA는 문서-단어 행렬을 기반으로 하므로, 단어의 순서나 문맥 정보를 고려하지 않는다. 이는 문맥을 중요시하는 텍스트 분석에서는 한계가 될 수 있다.<br>
 - 선형적 관계: LSA는 주로 선형적인 관계를 가정하기 때문에 비선형적인 의미 관계를 학습하는 데는 한계가 있다.<br>
 - 큰 데이터에서의 한계: 매우 큰 문서 집합에서는 SVD 계산이 복잡해지고 시간이 오래 걸릴 수 있다.<br>
▣ LSA의 응용분야
 - 정보 검색(IR, Information Retrieval): 검색 엔진에서 사용자가 입력한 질의(query)와 데이터베이스에 저장된 문서 간의 의미적 유사성을 분석하여, 더 관련성 높은 문서를 검색하는 데 LSA가 사용된다.<br>
 - 문서 분류 및 군집화: LSA는 문서 간의 잠재적인 의미적 유사성을 파악하여 문서들을 자동으로 분류하거나 군집화하는 데 사용된다.<br>
 - 추천 시스템: 사용자가 관심을 가질 만한 문서를 추천하는 시스템에서 LSA는 사용자와 문서 간의 잠재적인 의미적 관계를 찾아내는 데 도움을 준다.<br>
 - 주제 모델링(Topic Modeling): 문서 내에 숨겨진 주제(토픽)를 추출하는 데 LSA가 사용될 수 있다. 이는 LSA가 문서 내에서 자주 발생하는 단어 패턴을 학습하는 과정에서 주제를 식별할 수 있기 때문이다.<br>
 - 텍스트 유사도 계산: 두 문서가 의미적으로 얼마나 유사한지를 계산하는 데 LSA가 사용된다. 이는 중복 콘텐츠 탐지나 텍스트 유사도 기반 추천 시스템에서 유용하다.<br>
▣ LSA의 모델식 : 문서-용어 행렬 𝑋에 대해 SVD를 적용하여 차원을 축소(𝑈는 단어, 𝑉는 문서, Σ는 특이값 행렬)<br>
$X=UΣV^T$<br>

<br>

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 예시 문서
    docs = ["dog cat", "cat mouse", "dog mouse", "dog", "mouse"]

    # TF-IDF 행렬 생성
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)

    # LSA 적용
    svd = TruncatedSVD(n_components=2)
    lsa_result = svd.fit_transform(X)

    print(lsa_result)



<br>

# [8] 특이값 분해(Singular Value Decomposition, SVD)
SVD는 선형대수학에서 매우 중요한 행렬 분해 기법으로, 임의의 행렬을 세 개의 행렬로 분해하는 방식이다. SVD는 데이터 분석, 차원 축소, 신호 처리, 머신러닝 등 다양한 분야에서 활용되며, 특히 고차원의 데이터를 다룰 때 중요한 역할을 한다. SVD는 주로 행렬의 특이값과 특이벡터를 통해 행렬의 구조를 파악하고, 이를 통해 데이터의 패턴을 찾거나 압축하는 데 사용된다.<br>

SVD의 핵심개념
 - 모든 행렬에 적용 가능: SVD는 정방행렬뿐만 아니라 비정방행렬이나 비대칭 행렬에도 적용할 수 있다는 점에서 SVD는 다른 행렬 분해 기법들에 비해 범용성이 높다.<br>
 - 특이값: Σ의 대각 성분인 특이값은 A의 행렬에 대한 중요한 정보를 담고 있으며 특이값이 클수록 데이터의 변동성이 크다는 것을 의미하며, 작은 특이값은 데이터의 변동이 적은 방향을 나타낸다.<br>
 - 차원 축소: SVD는 특이값을 기준으로 가장 중요한 축을 선택하여 차원을 축소할 수 있다. 가장 큰 특이값에 해당하는 축만 남기고 나머지를 제거하면, 데이터의 중요한 정보는 유지하면서 노이즈나 불필요한 차원을 줄일 수 있다.<br>
SVD의 장점
 - 모든 행렬에 적용 가능: SVD는 정방, 비정방, 비대칭 행렬 등 어떤 형태의 행렬에도 적용할 수 있다.<br>
 - 차원 축소: 데이터를 저차원 공간으로 변환하면서도 중요한 패턴을 유지할 수 있다.<br>
 - 노이즈 제거: 데이터에서 노이즈를 제거하여 중요한 정보만 남길 수 있다.<br>
 - 추천 시스템에서 활용: 사용자-아이템 간의 관계를 분석하고, 효과적인 추천을 가능하게 한다.<br>
SVD의 단점
 - 계산 복잡도: SVD는 계산 비용이 매우 높으며, 특히 매우 큰 행렬의 경우 계산이 오래 걸릴 수 있다.<br>
 - 해석 어려움: 분해된 행렬들이 원본 데이터와 직관적인 관계를 가지지 않기 때문에, 결과를 해석하는 것이 어려울 수 있다.<br>
SVD의 응용분야
 - 차원 축소(Dimensionality Reduction) : 고차원 데이터를 저차원으로 변환할 때 SVD는 중요한 역할을 한다. 데이터에서 가장 중요한 패턴을 유지하면서도 차원을 줄일 수 있습니다. 예를 들어, 텍스트 데이터를 다루는 잠재 의미 분석(LSA)에서는 단어-문서 행렬을 SVD를 사용하여 차원을 축소하고, 문서 간의 의미적 관계를 파악한다.<br>
 - 데이터 압축(Data Compression) : 이미지, 영상, 신호 데이터와 같은 큰 데이터를 압축할 때도 SVD가 유용하다. 원래 데이터의 중요한 정보를 유지하면서 압축된 데이터를 생성할 수 있으며, 이를 통해 메모리 사용량을 줄이고 계산 시간을 단축할 수 있다.<br>
 - 노이즈 제거(Denoising) : 데이터에서 노이즈를 제거하는 데 SVD가 사용된다. 노이즈는 주로 작은 특이값에 해당하는 방향에서 발생하므로, 작은 특이값을 제거하고 큰 특이값에 해당하는 정보만 남기면 노이즈를 제거할 수 있다.<br>
 - 추천 시스템(Recommendation Systems) : 추천 시스템에서 사용자와 아이템 간의 상호작용 행렬을 SVD로 분해하여, 사용자에게 가장 적합한 아이템을 추천할 수 있다. 이는 영화 추천, 제품 추천 등 다양한 도메인에서 활용된다.<br>
 - 이미지 압축(Image Compression) : SVD는 이미지 데이터를 효율적으로 압축하는 데 사용된다. 이미지를 행렬로 표현한 후 SVD를 통해 차원을 축소하면, 원본 이미지의 중요한 정보는 유지하면서도 크기가 줄어든 이미지를 생성할 수 있다.<br> 

<br>
    import numpy as np
    from numpy.linalg import svd

    # 예시 데이터 생성
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # SVD 적용
    U, Sigma, VT = svd(A)

    print("U matrix:")
    print(U)
    print("Sigma values:")
    print(Sigma)
    print("VT matrix:")
    print(VT)

<br>



---
## 차원 축소 알고리즘 평가방법

**▣ 재구성 오류(Reconstruction Error) :** 차원 축소된 데이터를 원본 차원으로 복원하여 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)를 통해 재구성 오류를 계산

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)를 준비
    data = load_iris()
    X = data.data  # 입력 데이터 (특성)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)  # 차원 축소된 데이터

    # 재구성 오류 계산 : 차원 축소된 데이터를 원래 차원으로 복원하고 원본 데이터와의 평균 제곱 오차(MSE)를 계산
    X_reconstructed = pca.inverse_transform(X_reduced)  # 차원 축소 후 복원된 데이터
    reconstruction_error = mean_squared_error(X, X_reconstructed)  # 재구성 오류 계산
    print(f"Reconstruction Error (MSE): {reconstruction_error:.3f}")

<br>

**▣ 분산 유지율(Explained Variance Ratio) :** 각 주성분이 설명하는 분산 비율을 통해 데이터의 정보 손실 정도를 파악

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)를 준비합니다.
    data = load_iris()
    X = data.data  # 입력 데이터 (특성)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    pca.fit(X)  # PCA 학습

    # 분산 유지율 계산 : 각 주성분이 데이터의 분산을 얼마나 설명하는지 비율로 확인
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio per Component: {explained_variance_ratio}")
    print(f"Total Variance Retained: {sum(explained_variance_ratio):.3f}")  # 전체 분산 유지율

<br>

**▣ 상호 정보량(Mutual Information) :** 차원 축소 전후 데이터의 정보량을 비교

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)와 실제 레이블(y_true)를 준비
    data = load_iris()
    X = data.data         # 입력 데이터 (특성)
    y_true = data.target  # 실제 레이블 (클러스터링 평가 시 사용)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)  # 차원 축소된 데이터

    # KMeans를 사용하여 차원 축소된 데이터에서 클러스터링을 수행 : 클러스터 개수를 3으로 설정하여 실제 클래스 수와 맞추기
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X_reduced)  # 클러스터링 예측 레이블

    # 4. 상호 정보량 계산
    # (1) Adjusted Mutual Information (AMI) : 실제 레이블(y_true)과 클러스터링 예측 레이블(y_pred) 간의 유사도를 측정
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print(f"Adjusted Mutual Information (AMI): {ami:.3f}")

    # (2) Normalized Mutual Information (NMI) : 실제 레이블과 예측 레이블 간의 상호 정보량을 정규화하여 측정
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

<br>

**▣ 군집 평가 지표 :** 차원 축소 후 클러스터링을 수행하고 군집 평가 지표를 계산하여 차원 축소의 성능을 평가

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.model_selection import train_test_split

    # 데이터 로드
    data = load_iris()
    X = data.data
    y_true = data.target  # 실제 레이블 (평가를 위해 사용)

    # PCA를 사용하여 차원 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # KMeans를 사용하여 클러스터링 수행
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X_reduced)

    # 군집 평가 지표 계산
    # (1) Silhouette Score
    silhouette = silhouette_score(X_reduced, y_pred)
    print(f"Silhouette Score: {silhouette:.3f}")

    # (2) Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(X_reduced, y_pred)
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

    # (3) Adjusted Rand Index (ARI) - 실제 레이블과 예측 레이블 비교
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    # (4) Normalized Mutual Information (NMI) - 실제 레이블과 예측 레이블 비교
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

<br>

