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
    
    [연관 규칙 알고리즘 평가방법]
    ▣ 지지도(Support) : 특정 항목 집합이 전체 거래에서 얼마나 자주 나타나는지 나타낸다.
    ▣ 신뢰도(Confidence) : A가 주어졌을 때 B가 발생할 확률
    ▣ 향상도(Lift) : A와 B가 서로 독립적으로 발생하는 경우에 비해 A가 발생했을 때 B가 발생할 가능성이 얼마나 높은지를 나타낸다.
    ▣ 레버리지(Leverage) : A와 B의 결합 빈도가 두 항목이 독립적으로 발생하는 빈도와 얼마나 차이가 나는지 나타낸다.
    ▣ Conviction(확신도) : A가 발생할 때 B가 발생하지 않을 가능성이 독립적인 경우보다 얼마나 줄어드는지를 나타낸다.
    ▣ 컬러레이션 계수(Correlation Coefficient) : 0에 가까우면 두 항목 간에 상관관계가 없고, 양수나 음수로 갈수록 상관관계가 강하다.

    

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
    [9] SOM(Self-Organizing Maps)

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
▣ 단점 : 탐색 공간이 커지면 성능이 저하되고 대규모 데이터에서 비효율적일 수 있으며, 매번 새로운 후보집합 생성에 따른 큰 계산비용<br>
▣ 응용분야 : 시장 바구니 분석(장바구니 데이터에서 자주 함께 구매되는 상품을 찾음), 추천 시스템, 웹 페이지 연결성 분석<br>
▣ 모델식 : 지지도(Support): 특정 항목 집합이 전체 거래에서 발생하는 빈도, 신뢰도(Confidence): 특정 항목이 발생한 경우 다른 항목이 함께 발생할 확률, 향상도(Lift): 항목 간의 상호의존성을 측정<br>

    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    import matplotlib.pyplot as plt

    # 데이터 로드 및 전처리 (Online Retail 데이터 예제 사용)
    df = pd.read_excel("Online Retail.xlsx")
    df = df.dropna(subset=['InvoiceNo', 'Description'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[df['Country'] == 'United Kingdom']
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori 알고리즘
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # 결과 출력
    print("Apriori Frequent Itemsets:")
    print(frequent_itemsets)
    print("\nApriori Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # 시각화 - Lift vs Confidence
    plt.scatter(rules['confidence'], rules['lift'], alpha=0.5)
    plt.xlabel("Confidence")
    plt.ylabel("Lift")
    plt.title("Apriori Rules (Confidence vs Lift)")
    plt.show()

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

    from collections import defaultdict

    # 데이터 로드 및 전처리
    df = pd.read_excel("Online Retail.xlsx")
    df = df.dropna(subset=['InvoiceNo', 'Description'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[df['Country'] == 'United Kingdom']
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Eclat 알고리즘
    def eclat(data, min_support):
        transaction_index = defaultdict(set)
        for i, transaction in enumerate(data):
            for item in transaction:
                transaction_index[item].add(i)

        patterns = {frozenset([item]): trans for item, trans in transaction_index.items() if len(trans) >= min_support}
        return patterns

    # 트랜잭션 데이터로 변환
    transactions = basket.apply(lambda x: set(basket.columns[x == 1]), axis=1).tolist()
    min_support = 10
    patterns = eclat(transactions, min_support)

    # 결과 출력
    print("Eclat Frequent Patterns:")
    for pattern, transaction_ids in patterns.items():
        print(f"Pattern: {pattern}, Support: {len(transaction_ids)}")
    
<br>

# [4] Multi-level Association Rules
▣ 정의: 연관 규칙을 계층적으로 탐색하여 다중 수준에서 규칙을 생성하는 방식<br>
▣ 필요성: 제품 카테고리별 분석이 필요한 경우에 적합<br>
▣ 장점: 더 정교한 규칙을 생성<br>
▣ 단점: 복잡성이 증가하며, 해석이 어려워질 수 있음<br>
▣ 응용분야: 전자상거래, 추천 시스템, 마케팅 분석<br>

    # 예제 계층 구조 데이터 생성
    df['Category'] = df['Description'].apply(lambda x: 'Category1' if 'A' in x else 'Category2')

    # Apriori 알고리즘을 각 계층(Category)별로 적용
    rules_by_category = {}
    for category, group in df.groupby('Category'):
        basket = group.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        rules_by_category[category] = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # 결과 출력
    for category, rules in rules_by_category.items():
        print(f"\nCategory: {category} - Rules")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

<br>

# [5] Multi-dimensional Association Rules
▣ 정의: 여러 속성을 포함하여 다양한 차원의 규칙을 생성<br>
▣ 필요성: 다양한 속성 간 관계를 탐색하는 데 적합<br>
▣ 장점: 규칙의 범위를 확장할 수 있어 더 세밀한 규칙 도출 가능.<br>
▣ 단점: 복잡성과 해석의 어려움<br>
▣ 응용분야: 사용자 속성 기반 추천 시스템, 마케팅 인텔리전스<br>

    # Country 속성을 추가하여 Multi-dimensional Association Rules 생성
    df = df[df['Country'].isin(['United Kingdom', 'France'])]  # 특정 국가 데이터만 사용
    basket_sets = df.groupby(['InvoiceNo', 'Country', 'Description'])
    ['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    basket_sets = basket_sets.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori 알고리즘 적용
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print("Multi-dimensional Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

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
## [연관 규칙 알고리즘 평가방법]

    (공통소스)
    import pandas as pd

    # 예제 데이터 (거래 내역)
    data = {
        'milk': [1, 0, 1, 1, 0],
        'bread': [1, 1, 1, 0, 1],
        'butter': [0, 1, 0, 1, 0],
        'beer': [1, 1, 1, 0, 0]}
    df = pd.DataFrame(data)

    # 전체 거래 수
    total_transactions = len(df)

    # 특정 항목 집합 A와 B 설정 : A (milk) => B (bread)
    A = 'milk'
    B = 'bread'

    # A와 B가 동시에 발생한 거래 수
    support_AB_count = len(df[(df[A] == 1) & (df[B] == 1)])

    # A가 발생한 거래 수
    support_A_count = len(df[df[A] == 1])

    # B가 발생한 거래 수
    support_B_count = len(df[df[B] == 1])

<br>

**▣ 지지도(Support):** 특정 항목 집합이 전체 거래에서 얼마나 자주 나타나는지 나타낸다.<br>
Support(A) = (거래에서 A가 발생한 횟수)/(전체 거래 수)<br>

    # Support (A와 B가 동시에 발생한 비율)
    support_AB = support_AB_count / total_transactions
    print(f"Support(A => B): {support_AB:.3f}")

<br>

**▣ 신뢰도(Confidence):** A가 주어졌을 때 B가 발생할 확률<br>
Confidence(A ⇒ B) = Support(A ∩ B)/Support(A)<br>

    # Confidence (A가 발생했을 때 B가 발생할 확률)
    confidence_A_B = support_AB_count / support_A_count if support_A_count > 0 else 0
    print(f"Confidence(A => B): {confidence_A_B:.3f}")

<br>

**▣ 향상도(Lift):** A와 B가 서로 독립적으로 발생하는 경우에 비해 A가 발생했을 때 B가 발생할 가능성이 얼마나 높은지를 나타낸다. 1이면 두 항목이 독립적, 1보다 크면 양의 상관관계, 1보다 작으면 음의 상관관계<br>
Lift(A ⇒ B) = Confidence(A ⇒ B)/Support(B)<br>

    # Lift (A와 B가 독립적인 경우에 비해 A가 발생할 때 B가 발생할 가능성 비율)
    lift_A_B = confidence_A_B / (support_B_count / total_transactions) if support_B_count > 0 else 0
    print(f"Lift(A => B): {lift_A_B:.3f}")

<br>

**▣ 레버리지(Leverage):** A와 B의 결합 빈도가 두 항목이 독립적으로 발생하는 빈도와 얼마나 차이가 나는지 나타낸다. 0이면 두 항목이 독립적<br>
Leverage(A ⇒ B) =  Support(A ∩ B) - (Support(A) × Support(B))<br>

    # Leverage (A와 B가 독립적일 때 결합 빈도와의 차이)
    leverage_A_B = support_AB - (support_A_count / total_transactions) * (support_B_count / total_transactions)
    print(f"Leverage(A => B): {leverage_A_B:.3f}")

<br>

**▣ Conviction(확신도):** A가 발생할 때 B가 발생하지 않을 가능성이 독립적인 경우보다 얼마나 줄어드는지를 나타낸다. 1에 가까우면 A와 B는 서로 독립적<br>
Conviction(A ⇒ B) = (1-Support(B))/(1-Confidence(A ⇒ B))<br>

    # Conviction (A가 발생했을 때 B가 발생하지 않을 가능성 감소율)
    conviction_A_B = (1 - (support_B_count / total_transactions)) / (1 - confidence_A_B) if confidence_A_B < 1 else float('inf')
    print(f"Conviction(A => B): {conviction_A_B:.3f}")

<br>

**▣ 컬러레이션 계수(Correlation Coefficient):** 0에 가까우면 두 항목 간에 상관관계가 없고, 양수나 음수로 갈수록 상관관계가 강하다.<br>

    # Correlation Coefficient (피어슨 상관계수)
    from scipy.stats import pearsonr

    correlation_A_B, _ = pearsonr(df[A], df[B])
    print(f"Correlation Coefficient(A, B): {correlation_A_B:.3f}")

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

# [3] UMAP(Uniform Manifold Approximation and Projection)
▣ 정의: 데이터의 국소 구조와 전역 구조를 동시에 보존하면서 저차원으로 투영하는 비선형 차원 축소 알고리즘<br>
▣ 필요성: 고차원 데이터를 저차원에서 시각화하면서 데이터의 전체적 및 국소적 관계를 동시에 보존하기 위해 사용<br>
▣ 장점: t-SNE보다 계산이 빠르고, 대규모 데이터에서도 잘 작동, 데이터의 전역적 및 국소적 구조를 동시에 보존<br>
▣ 단점: 일부 매개변수 조정이 필요하며, 결과가 매개변수에 민감할 수 있음<br>
▣ 응용분야: 대용량 데이터 시각화, 생물정보학, 텍스트 분석 등<br>
▣ 모델식: 이론적으로는 리만 거리와 초구 면적 개념을 이용하여 데이터의 근접성을 유지하면서 고차원에서 저차원으로 투영<br>

    import umap
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # UMAP 적용
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data.target)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("UMAP on Iris Dataset")
    plt.colorbar()
    plt.show()

<br>

# [4] SVD(Singular Value Decomposition)
▣ 정의: 임의의 행렬을 세 개의 행렬로 분해하는 방식으로 행렬의 특이값과 특이벡터를 통해 행렬의 구조를 파악하고, 이를 통해 데이터의 패턴을 찾거나 압축하는 데 사용<br> 
▣ 장점 : 정방/비정방/비대칭 행렬 등 어떤 형태의 행렬에도 적용 가능, 데이터를 저차원 공간으로 변환하면서도 중요한 패턴을 유지, 데이터에서 노이즈를 제거하여 중요한 정보만 남길 수 있음<br>
▣ 단점 : 특히 매우 큰 행렬의 경우 계산이 오래 걸릴 수 있으며, 분해된 행렬들이 원본 데이터와 직관적인 관계를 가지지 않기 때문에 결과를 해석하는 것이 어려울 수 있음<br>
▣ 응용분야 : 단어-문서 행렬의 차원 축소, 데이터 압축, 노이즈 제거, 추천 시스템, 이미지 압축<br> 
▣ 모델식 : $X=UΣV^T$<br>
𝑋는 𝑚×𝑛 크기의 원본 행렬, 𝑈는 𝑚×𝑚 크기의 좌측 직교 행렬, Σ는 𝑚×𝑛 크기의 대각 행렬로 특이값이 대각 원소로 배치, $𝑉^𝑇$는 𝑛×𝑛 크기의 우측 직교 행렬<br>

<br>

    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # SVD 적용
    svd = TruncatedSVD(n_components=2)
    X_svd = svd.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_svd[:, 0], X_svd[:, 1], c=data.target)
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("SVD on Iris Dataset")
    plt.colorbar()
    plt.show()

    # 분산 유지율 출력
    print("Explained Variance Ratio:", svd.explained_variance_ratio_)
    print("Total Variance Retained:", sum(svd.explained_variance_ratio_))

<br>

# [5] ICA(Independent Component Analysis)
▣ 정의 : 다변량 신호에서 통계적으로 독립적인 성분을 추출하는 비선형 차원 축소 기법. PCA는 데이터의 분산을 최대화하는 축을 찾는 반면, ICA는 신호 간의 독립성을 기반으로 성분을 찾는다. 또한 PCA는 가우시안 분포를 가정하고 데이터의 상관관계만을 이용해 차원을 축소하거나 성분을 찾는 반면, ICA는 신호들 간의 고차원적 통계적 독립성에 초점을 맞추기 때문에 더 복잡한 구조의 신호분리 문제를 해결<br>
▣ 필요성 : 관측된 신호가 여러 독립적인 원천 신호의 혼합으로 구성될 때 각 독립적인 신호를 복원하는 데 필요하며 특히 신호 처리 및 음성 분리에 유용<br>
▣ 응용분야 : 뇌파(EEG) 신호 분석, 음성 신호 분리, 이미지 처리<br>
▣ 장점 : 통계적으로 독립적인 신호를 분리할 수 있으며 신호 처리, 이미지 분할, 음성 분리 등에서 강력한 성능을 발휘<br>
▣ 단점 : 잡음에 민감하고, 원래 신호의 순서를 보장하지 않으며, 성분의 크기도 원래 신호와 다를 수 있어서 추가적인 후처리가 필요<br>
▣ 모델식 : 관측 데이터 𝑋=𝐴𝑆에서 𝐴는 혼합 행렬, 𝑆는 독립 성분 행렬이며, 𝐴와 𝑆를 추정하여 𝑆를 추출<br>
▣ 알고리즘 : 비선형성을 이용해 독립 성분을 빠르게 찾는 방법으로 신호의 비정규성을 최대화하는 방향으로 성분을 추정하는 Fast ICA과 정보 이론을 기반으로 한 방법으로, 관측된 데이터에서 정보량을 최대화하는 방식으로 독립 성분을 추정하는 Infomax ICA<br>

<br>

    import numpy as np
    from sklearn.decomposition import FastICA
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # ICA 적용
    ica = FastICA(n_components=2, random_state=42)
    X_ica = ica.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_ica[:, 0], X_ica[:, 1], c=data.target)
    plt.xlabel("ICA Component 1")
    plt.ylabel("ICA Component 2")
    plt.title("ICA on Iris Dataset")
    plt.colorbar()
    plt.show()

<br> 

# [6] LDA(Linear Discriminant Analysis)
▣ 정의: 클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하는 선형 차원 축소 기법으로 주로 지도 학습에서 사용<br>
▣ 필요성: 클래스 간 분리를 극대화하면서 데이터를 저차원으로 투영하여 분류 문제의 성능을 향상시키기 위해 필요<br>
▣ 장점: 클래스 분리를 극대화하여 분류 성능을 개선할 수 있으며, 선형 변환을 통해 효율적으로 차원을 축소<br>
▣ 단점: 데이터가 선형적으로 구분되지 않는 경우 성능이 저하될 수 있으며, 클래스 간 분포가 정규 분포를 따를 때 더 효과적<br>
▣ 응용분야: 얼굴 인식, 이미지 분류, 텍스트 분류 등<br>
▣ 모델식: 두 클래스 간의 분산 비율을 최대화하는 방향으로 데이터를 투영<br>

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data
    y = data.target

    # LDA 적용
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    # 결과 시각화
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.title("LDA on Iris Dataset")
    plt.colorbar()
    plt.show()

<br>

# [7] Isomap
▣ 정의: 데이터의 기하학적 구조를 보존하여 고차원 데이터를 저차원으로 투영하는 비선형 차원 축소 기법<br>
▣ 필요성: 비선형적인 데이터 구조를 저차원에서도 유지하며 시각화할 때 유용<br>
▣ 장점: 고차원 데이터의 매니폴드(저차원 다양체) 구조를 잘 보존하며, 국소적인 거리 정보를 기반으로 데이터의 구조를 유지<br>
▣ 단점: 데이터가 고차원에서 매니폴드 구조를 형성하지 않는 경우 효과적이지 않으며, 계산 비용이 높아 대용량 데이터에는 부적합<br>
▣ 응용분야: 시각화, 이미지 및 텍스트 데이터 분석, 생물정보학<br>
▣ 모델식: 근접 그래프와 다차원 척도를 결합하여 비선형 구조를 보존<br>

    from sklearn.manifold import Isomap
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # Isomap 적용
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=data.target)
    plt.xlabel("Isomap Component 1")
    plt.ylabel("Isomap Component 2")
    plt.title("Isomap on Iris Dataset")
    plt.colorbar()
    plt.show()

<br>

# [8] MDS(Multidimensional Scaling)
▣ 정의: MDS는 고차원 데이터 포인트 간의 거리를 보존하며 저차원으로 투영하는 차원 축소 기법<br>
▣ 필요성: 데이터의 유사성 또는 거리 정보를 저차원에서도 유지하여 시각화하기 위해 사용<br>
▣ 장점: 거리 정보를 보존하므로 데이터의 기하학적 관계를 잘 유지하며, 비선형 구조를 일부 보존<br>
▣ 단점: 계산 비용이 높고, 대용량 데이터에는 적합하지 않으며, 초기화에 민감하여 결과가 다를 수 있음<br>
▣ 응용분야: 심리학, 생물정보학, 마케팅 데이터 분석 등<br>
▣ 모델식: 데이터 포인트 간의 거리 행렬을 유지하며 저차원에서 구성<br>

    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # 데이터 로드
    data = load_iris()
    X = data.data

    # MDS 적용
    mds = MDS(n_components=2, random_state=42)
    X_mds = mds.fit_transform(X)

    # 결과 시각화
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=data.target)
    plt.xlabel("MDS Component 1")
    plt.ylabel("MDS Component 2")
    plt.title("MDS on Iris Dataset")
    plt.colorbar()
    plt.show()

<br>


# [9] SOM(Self-Organizing Maps)
▣ 정의 : 고차원의 데이터를 저차원(일반적으로 2차원) 공간으로 투영하여 데이터의 구조를 시각화하는 데 사용. PCA는 선형 변환을 통해 차원 축소를 수행하지만, SOM은 비선형 변환을 사용하여 더 복잡한 데이터 구조를 반영할 수 있으며, k-평균은 각 군집의 중심을 찾는 방식으로 군집화를 수행하는 반면, SOM은 뉴런이 격자 형태로 조직되어 있어 더 직관적인 시각화가 가능<br> 
▣ 절차
(1) 초기화: SOM의 각 뉴런에 임의의 가중치 벡터를 할당(이 가중치 벡터는 입력 데이터와 같은 차원)<br>
(2) 입력 데이터 선택: 학습 과정에서 입력 데이터 벡터 하나를 무작위로 선택<br>
(3) 승자 뉴런(BMU, Best Matching Unit) 찾기: SOM의 모든 뉴런 중에서 현재 입력 벡터와 가장 유사한(가중치 벡터 간의 유클리드 거리로 측정) 뉴런을 찾는 경쟁 학습의 핵심 단계<br>
(4) 가중치 벡터 갱신: 선택된 승자 뉴런과 그 주변 이웃 뉴런들의 가중치 벡터를 조정한다. 이때, 가중치 벡터는 입력 데이터에 더 가깝게 이동<br> 
▣ 장점 : 데이터에 대한 사전 정보가 없어도 유용하게 사용 가능, 군집의 분포나 데이터의 경향성을 직관적으로 이해, 입력 데이터의 이웃 관계를 보존하면서 저차원으로 투영하므로 원래 데이터의 공간적 관계를 유지<br> 
▣ 단점 : 학습률과 이웃 크기 등 여러 파라미터를 적절히 설정해야 하며, 대규모 데이터 학습에 비효율적, 변환된 맵을 해석하는 것이 PCA 등의 선형 변환보다 더 어려움<br> 
▣ 응용분야 : 이미지 분석, 문서 분류, 음성 인식, 생물정보학<br>
▣ 모델식 : 뉴런의 위치 𝑟와 입력 벡터 𝑥 간의 거리 함수로 클러스터를 형성(𝜂(𝑡)는 학습률, ℎ(𝑡)는 이웃 함수)<br>
$W(t+1)=W(t)+\theta(t)\cdot\eta(t)\cdot(X-W(t))$<br>

    !pip install minisom

    from minisom import MiniSom
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import MinMaxScaler

    # 데이터 로드 및 정규화
    data = load_iris()
    X = data.data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # SOM 초기화 및 학습
    som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(X_scaled, 100)  # 100회 반복 학습

    # SOM 시각화
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(X_scaled):
        w = som.winner(x)
        plt.text(w[0] + 0.5, w[1] + 0.5, str(data.target[i]),
        color=plt.cm.rainbow(data.target[i] / 2.0),
        fontdict={'weight': 'bold', 'size': 11})

    plt.title("SOM Clustering of Iris Data")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.grid()
    plt.show()

<br> 

---
## [차원 축소 알고리즘 평가방법]

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

