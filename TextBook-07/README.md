#  07-1 : 군집 평가 지표

---
	
	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합

	  
---


## ▣ 평가지표 수식

| 지표명 | 의미 | 수식(대표 정의) |
|---|---|---|
| **[1] Silhouette Coefficient** | 한 점이 자기 군집 평균거리 $a(i)$ 대비 가장 가까운 다른 군집 평균거리 $b(i)$ 로 분리도 측정 | $s(i)=\dfrac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}$<br>$\bar{s}=\dfrac{1}{n}\sum_{i=1}^{n}s(i)$ |
| **[2] Davies–Bouldin Index (DBI)** | 군집 내 응집도 대비 군집 간 분리도 | $DBI=\dfrac{1}{K}\sum_{i=1}^{K}\max_{j\ne i}\dfrac{S_i+S_j}{M_{ij}}$<br>$S_i=\dfrac{1}{\lvert C_i\rvert}\sum_{x\in C_i}\lVert x-\mu_i\rVert,\; M_{ij}=\lVert\mu_i-\mu_j\rVert$ |
| **[3] Dunn Index (DI)** | 가장 가까운 군집 간 최소거리 대비 최대 군집 지름 | $DI=\dfrac{\min_{i\ne j}\,\delta(C_i,C_j)}{\max_k\,\Delta(C_k)}$<br>$\delta$: 군집 간 최소거리, $\Delta$: 군집 지름(내부 최대거리) |
| **[4] Calinski–Harabasz Index (CHI)** | 군집 사이 분산 / 군집 내 분산 | $CH=\dfrac{\mathrm{Tr}(B_K)/(K-1)}{\mathrm{Tr}(W_K)/(n-K)}$<br>$\mathrm{Tr}(B_K)=\sum_k \lvert C_k\rvert\,\lVert\mu_k-\bar{x}\rVert^2,\;\mathrm{Tr}(W_K)=\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$ |
| **[5] Within-Cluster Sum of Squares (WCSS)** | 각 점이 군집 중심까지의 제곱거리 합 K-means 목적함수와 동일 | $\mathrm{WCSS}=\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k\rVert^{2}$ |


<br>


## ▣ 평가지표 결과해석

| 지표                                | 목표                    | 권장 해석 기준(경험치)                                                           | 비고                                                             |
| --------------------------------- | --------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------- |
| **[1] Silhouette Coefficient**        | ✓ 높을수록 좋음 (범위 −1~1)   | **≥ 0.70 매우 좋음**, **0.50~0.69 양호**, **0.25~0.49 보통**, **< 0.25 미흡**     | 군집 내 응집 vs 인접 군집과 분리. 평균값과 군집별 분포를 함께 확인 권장.               |
| **[2] Davies–Bouldin Index (DBI)**    | ✗ 낮을수록 좋음 (하한 0)      | **≤ 0.50 매우 좋음**, **0.51~0.99 양호**, **1.00~1.49 보통**, **≥ 1.50 미흡**     | 군집 응집도 대비 중심 간 분리. 스케일·거리척도에 민감.                           |
| **[3] Dunn Index (DI)**               | ✓ 높을수록 좋음 (상한 데이터 의존) | **≥ 0.50 매우 좋음**, **0.30~0.49 양호**, **0.10~0.29 보통**, **< 0.10 미흡** | 최소 군집 간 거리 / 최대 지름이라 값이 전반적으로 작게 나오는 편. 이상치·밀도 차이에 민감. |
| **[4] Calinski–Harabasz Index (CHI)** | ✓ 높을수록 좋음             | **절대 임계치 없음** → 동일 데이터에서 k 간 상대 비교: 국소 최대/엘보우 지점이 바람직           | 데이터/스케일에 강하게 의존. 보통 k 스윕으로 비교.                             |
| **[5] Within-Cluster Sum of Squares (WCSS)**                | ✗ 낮을수록 좋음             | **절대 임계치 없음** → 엘보우 지점에서 k 선택                                 | 표준화 여부·특징 개수에 좌우. k↑→단조감소이므로 상대 비교 전용.                 |



<br>

### Iris 데이터 + K-means(k=3) 학습 후 평가지표 5종 출력 소스

**[1] Silhouette Coefficient**
<br>
**[2] Davies-Bouldin Index(DBI)**
<br>
**[3] Dunn Index(DI)** scikit-learn 미제공으로 커스텀 구현
<br>
**[4] Calinski-Harabasz Index(CHI)**
<br>
**[5] Within-Cluster Sum of Squares(WCSS = inertia)**
<br>


	# ---------- 경고 방지용 환경변수: 반드시 상단에서 설정 ----------
	import os
	os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP 스레드
	os.environ["MKL_NUM_THREADS"] = "1"       # MKL 스레드
	os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS 스레드
	os.environ["NUMEXPR_NUM_THREADS"] = "1"   # numexpr 스레드
	# 물리 코어 수 직접 지정하고 싶다면 주석 해제 (예: 8코어)
	# os.environ["LOKY_MAX_CPU_COUNT"] = "8"
	
	# ------------------------- 라이브러리 임포트 -------------------------
	import numpy as np
	from sklearn import datasets
	from sklearn.cluster import KMeans
	from sklearn.metrics import (
	    silhouette_score,
	    davies_bouldin_score,
	    calinski_harabasz_score,
	)
	from sklearn.metrics import pairwise_distances
	
	
	# ------------------------- Dunn Index (커스텀) -------------------------
	def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
	    """
	    Dunn Index (값이 클수록 좋음)
	    DI = (군집 간 최소거리) / (군집 내 최대 지름)
	    - 단일 샘플 군집이 있거나, 분모가 0이면 NaN 반환
	    """
	    unique_labels = np.unique(labels)
	    clusters = [X[labels == c] for c in unique_labels]
	
	    if len(clusters) < 2:
	        return float("nan")
	
	    # 군집 내 최대 지름(최대 쌍거리)
	    intra_max = 0.0
	    for c in clusters:
	        if len(c) <= 1:
	            # 단일 포인트 군집은 지름이 정의 어려움 → 넘어가되 전체 판단에 영향
	            continue
	        d = pairwise_distances(c)
	        intra_max = max(intra_max, float(np.max(d)))
	
	    # 군집 간 최소거리(단일 링크)
	    inter_min = np.inf
	    for i in range(len(clusters)):
	        for j in range(i + 1, len(clusters)):
	            if len(clusters[i]) == 0 or len(clusters[j]) == 0:
	                continue
	            d = pairwise_distances(clusters[i], clusters[j])
	            inter_min = min(inter_min, float(np.min(d)))
	
	    if intra_max == 0.0 or not np.isfinite(inter_min):
	        return float("nan")
	    return inter_min / intra_max
	
	
	# ------------------------- 메인 루틴 -------------------------
	def main(k: int = 3, random_state: int = 42) -> None:
	    # 1) 데이터 로드
	    iris = datasets.load_iris()
	    X = iris.data
	
	    # 2) K-means 학습
	    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
	    labels = km.fit_predict(X)
	
	    # 3) 지표 계산
	    silhouette = float(silhouette_score(X, labels))
	    dbi = float(davies_bouldin_score(X, labels))
	    dunn = float(dunn_index(X, labels))
	    chi = float(calinski_harabasz_score(X, labels))
	    wcss = float(km.inertia_)  # Within-Cluster Sum of Squares
	
	    # 4) 출력
	    print(f"=== Iris + K-means (k={k}) ===")
	    print(f"[1] Silhouette Coefficient : {silhouette:.4f}")
	    print(f"[2] Davies-Bouldin Index   : {dbi:.4f}")
	    print(f"[3] Dunn Index             : {dunn:.4f}")
	    print(f"[4] Calinski-Harabasz     : {chi:.4f}")
	    print(f"[5] WCSS (Inertia)        : {wcss:.4f}")
	
	
	if __name__ == "__main__":
	    main(k=3)



### (소스 실행 결과)

	=== Iris + K-means (k=3) ===
	[1] Silhouette Coefficient : 0.5528
	[2] Davies-Bouldin Index   : 0.6620
	[3] Dunn Index             : 0.0988
	[4] Calinski-Harabasz     : 561.6278
	[5] WCSS (Inertia)        : 78.8514


### (결과 분석)

	[1] Silhouette Coefficient : 0.5528		=> 0.5 이상이면 양호(군집 내 응집도와 군집 간 분리가 꽤 확보됨)
	[2] Davies-Bouldin Index   : 0.6620		=> 1 미만이면 보통(분리도/응집도 균형이 준수)
	[3] Dunn Index             : 0.0988		=> 1 미만이면 미흡(군집 간 최소거리가 작게 잡혀 낮게 나왔을 가능성)
	[4] Calinski-Harabasz     : 561.6278	=> k=3이 꽤 설득력 있음
	[5] WCSS (Inertia)        : 78.8514		=> 응집도 양호(단독 평가는 어렵고, k를 바꿔 엘보우로 비교 추천)

---

#  07-2 : 연관규칙 평가 지표

---
	
	[1] 지지도(Support) : 특정 항목 집합이 전체 거래에서 얼마나 자주 나타나는지 나타낸다.
	[2] 신뢰도(Confidence) : A가 주어졌을 때 B가 발생할 확률
	[3] 향상도(Lift) : A와 B가 서로 독립적으로 발생하는 경우에 비해 A가 발생했을 때 B가 발생할 가능성이 얼마나 높은지를 나타낸다.
	[4] 레버리지(Leverage) : A와 B의 결합 빈도가 두 항목이 독립적으로 발생하는 빈도와 얼마나 차이가 나는지 나타낸다.
	[5] Conviction(확신도) : A가 발생할 때 B가 발생하지 않을 가능성이 독립적인 경우보다 얼마나 줄어드는지를 나타낸다.
	[6] 상관계수(Correlation Coefficient)는 두 변수 간의 관계의 강도와 방향

	  
---

## [연관 규칙 알고리즘 평가방법]

**▣ 지지도(Support):** 특정 항목 집합이 전체 거래에서 얼마나 자주 나타나는지 나타낸다.<br>
Support(A) = (거래에서 A가 발생한 횟수)/(전체 거래 수)<br>

**▣ 신뢰도(Confidence):** A가 주어졌을 때 B가 발생할 확률<br>
Confidence(A ⇒ B) = Support(A ∩ B)/Support(A)<br>

**▣ 향상도(Lift):** A와 B가 서로 독립적으로 발생하는 경우에 비해 A가 발생했을 때 B가 발생할 가능성이 얼마나 높은지를 나타낸다. 1이면 두 항목이 독립적, 1보다 크면 양의 상관관계, 1보다 작으면 음의 상관관계<br>
Lift(A ⇒ B) = Confidence(A ⇒ B)/Support(B)<br>

**▣ 레버리지(Leverage):** A와 B의 결합 빈도가 두 항목이 독립적으로 발생하는 빈도와 얼마나 차이가 나는지 나타낸다. 0이면 두 항목이 독립적<br>
Leverage(A ⇒ B) =  Support(A ∩ B) - (Support(A) × Support(B))<br>

**▣ Conviction(확신도):** A가 발생할 때 B가 발생하지 않을 가능성이 독립적인 경우보다 얼마나 줄어드는지를 나타낸다. 1에 가까우면 A와 B는 서로 독립적<br>
Conviction(A ⇒ B) = (1-Support(B))/(1-Confidence(A ⇒ B))<br>

**▣ 상관계수(Correlation Coefficient):** 0에 가까우면 두 항목 간에 상관관계가 없고, 양수나 음수로 갈수록 상관관계가 강하다.<br>

<br>

---

#  07-3 : 차원축소 평가 지표

---
	
	▣ 재구성 기반 : 원본 복원 능력
	[1] 재구성 오류(Reconstruction Error) : 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)
	[2] 분산 유지율(Explained Variance Ratio) : 각 주성분이 설명하는 분산 비율로 데이터의 정보 손실정도 파악
	[3] 상호 정보량(Mutual Information) :  차원 축소 전후 데이터의 정보량을 비교

	▣ 구조 보존 기반 : 거리·이웃 관계 유지
	[4] 근접도 보존 : Trustworthiness, Continuity
	[5] 거리/유사도 보존 : Stress, Sammon Error
	[6] 지역/전역구조 : LCMC(Local Continuity Meta Criterion)
	[7] 쌍의 상관계수 : Spearman’s ρ

	▣ 활용 성능 기반 : 축소된 표현의 유용성
	[8] Silhouette Score
	[9] Davies-Bouldin Index (DBI)
	[10] Adjusted Rand Index (ARI)
	[11] Normalized Mutual Information (NMI)
	  
---


