
#  07-1 : 군집 평가지표

	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합
	  

#  07-2 : 연관규칙 평가지표

	[1] 지지도(Support)
	[2] 신뢰도(Confidence)
	[3] 향상도(Lift)
	[4] 레버리지(Leverage)
	[5] 확신도(Conviction)
	[6] 상관계수(Correlation Coefficient)
	  
#  07-3 : 차원축소 평가지표

	[1] 재구성 오류(Reconstruction Error) : 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)
	[2] 분산 유지율(Explained Variance Ratio) : 각 주성분이 설명하는 분산 비율로 데이터의 정보 손실정도 파악
	[3] 상호 정보량(Mutual Information) :  차원 축소 전후 데이터의 정보량을 비교
	[4] 근접도 보존 : Trustworthiness, Continuity
	[5] 거리/유사도 보존 : Stress, Sammon Error
	[6] 지역/전역구조 : LCMC(Local Continuity Meta Criterion)
	[7] 쌍의 상관계수 : Spearman’s ρ
	[8] Silhouette Score
	[9] Davies-Bouldin Index (DBI)
	[10] Adjusted Rand Index (ARI)
	[11] Normalized Mutual Information (NMI)
	  


---

#  07-1 : 군집 평가지표

---

	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합
	  
---


## ▣ 군집 평가지표 수식

| 지표 | 의미 | 수식 |
|---|---|---|
| **[1] Silhouette Coefficient** | 한 점이 자기 군집 평균거리 $a(i)$ 대비 가장 가까운 다른 군집 평균거리 $b(i)$ 로 분리도 측정 | $s(i)=\dfrac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}$<br>$\bar{s}=\dfrac{1}{n}\sum_{i=1}^{n}s(i)$ |
| **[2] Davies–Bouldin Index (DBI)** | 군집 내 응집도 대비 군집 간 분리도 | $DBI=\dfrac{1}{K}\sum_{i=1}^{K}\max_{j\ne i}\dfrac{S_i+S_j}{M_{ij}}$<br>$S_i=\dfrac{1}{\lvert C_i\rvert}\sum_{x\in C_i}\lVert x-\mu_i\rVert,\; M_{ij}=\lVert\mu_i-\mu_j\rVert$ |
| **[3] Dunn Index (DI)** | 가장 가까운 군집 간 최소거리 대비 최대 군집 지름 | $DI=\dfrac{\min_{i\ne j}\,\delta(C_i,C_j)}{\max_k\,\Delta(C_k)}$<br>$\delta$: 군집 간 최소거리, $\Delta$: 군집 지름(내부 최대거리) |
| **[4] Calinski–Harabasz Index (CHI)** | 군집 사이 분산 / 군집 내 분산 | $CH=\dfrac{\mathrm{Tr}(B_K)/(K-1)}{\mathrm{Tr}(W_K)/(n-K)}$<br>$\mathrm{Tr}(B_K)=\sum_k \lvert C_k\rvert\,\lVert\mu_k-\bar{x}\rVert^2,\;\mathrm{Tr}(W_K)=\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$ |
| **[5] Within-Cluster Sum of Squares (WCSS)** | 각 점이 군집 중심까지의 제곱거리 합 K-means 목적함수와 동일 | $\mathrm{WCSS}=\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k\rVert^{2}$ |


<br>


## ▣ 군집 평가지표 결과해석

| 지표                                | 목표                    | 권장 해석 기준                                                           | 비고                                                             |
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

#  07-2 : 연관규칙 평가지표

---
	
	[1] 지지도(Support)
	[2] 신뢰도(Confidence)
	[3] 향상도(Lift)
	[4] 레버리지(Leverage)
	[5] 확신도(Conviction)
	[6] 상관계수(Correlation Coefficient)
	  
---

## ▣ 연관규칙 평가지표 수식
           
| 지표                       | 의미                                         | 수식                                                             |
| ------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| **[1] 지지도(Support)**         | A와 B가 동시에 발생할 비율                       | `support(A→B) = P(A ∧ B) = count(A ∧ B) / N`                           |
| **[2] 신뢰도(Confidence)**      | A가 발생했을 때 B가 함께 발생할 조건부 확률             | `confidence(A→B) = P(B ∣ A) = P(A ∧ B) / P(A)`                         |
| **[3] 향상도(Lift)**            | 독립 가정 대비 연관 강도                         | `lift(A→B) = P(A ∧ B) / ( P(A) · P(B) ) = confidence(A→B) / P(B)`      |
| **[4] 레버리지(Leverage)**       | 실제 동시발생과 기대 동시발생의 차이                   | `leverage(A,B) = P(A ∧ B) − P(A)·P(B)`                                 |
| **[5] 확신도(Conviction)**      | 규칙이 없을 때의 B 부정 확률 대비, 규칙 하의 오류율 감소 | `conviction(A→B) = (1 − P(B)) / (1 − confidence(A→B))`                 |
| **[6] 상관계수(Correlation, φ)** | A–B 이진 상관(피어슨 φ)                       | `φ(A,B) = ( P(A ∧ B) − P(A)·P(B) ) / √( P(A)(1−P(A)) · P(B)(1−P(B)) )` |


<br>


## ▣ 연관규칙 평가지표 결과해석

| 지표                       | 목표                               | 권장 해석 기준                                                              | 비고                                                              |
| ------------------------- | -------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **[1] 지지도(Support)**         | 높을수록 좋음(빈도·보편성)                  | **≥ 5%**: 매우 흔함, **1~5%**: 유의미, **< 1%**: 희귀                    | 데이터 크기·품목 수에 크게 의존. 너무 낮으면 우연 규칙 다수 → min_support 설정 필수.    |
| **[2] 신뢰도(Confidence)**      | 높을수록 좋음(조건부 정확도)                 | **≥ 0.80**: 강함, **0.60~0.79**: 양호, **< 0.60**: 약함                        | 빈발 후건(B)에 끌려 과대평가될 수 있음. Lift/Leverage와 함께 보정 필요.       |
| **[3] 향상도(Lift)**            | 1보다 클수록 좋음(독립 대비 강화)             | **> 2.0**: 강함, **1.2~2.0**: 유의미, **≈1.0**: 독립, **< 1.0**: 음의 연관      | 희귀 아이템 조합에서 과대될 수 있음 → 최소 지지도와 함께 해석. 도메인 검증 필수.            |
| **[4] 레버리지(Leverage)**       | 0에서 멀수록 강함(양수=양의 연관)             | **> 0.05**: 강함, **0.01~0.05**: 유의미, **≈ 0**: 독립                       | 절대적 “추가 동시발생 비율”. 빈도 민감도가 낮아 lift 보완에 유용. 표본 커질수록 작은 값도 의미. |
| **[5] 확신도(Conviction)**      | 1보다 클수록 좋음(규칙 위반률 감소)            | **> 2.0**: 강함, **1.2~2.0**: 유의미, **≈1.0**: 독립                            | confidence=1이고 P(B)<1이면 ∞. 후건이 드문 규칙에서 왜곡 가능.       |
| **[6] 상관계수(Correlation, φ)** | **∣φ∣**(=abs(φ)) 클수록 강함 · 부호로 방향 | **≥ 0.5**: 강함, **0.3~0.49**: 중간, **0.1–0.29**: 약함, **< 0.1**: 매우 약함 | 대칭 지표(순서 무관). 음수(φ<0)는 상호 배타 성향 의미. 빈도·희소성 영향은 중간 정도.       |



---

#  07-3 : 차원축소 평가지표

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

## 평가지표 수식

| 지표명 | 수식 | 설명 |
|---------|------|------|
| **(1) 재구성 오류 (Reconstruction Error)** | ![](https://latex.codecogs.com/svg.image?RE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5C%7C%20x_i%20-%20%5Chat%7Bx%7D_i%20%5C%7C%5E2) | 원본 데이터 \(x_i\)와 복원된 데이터 \(\hat{x}_i\)의 평균제곱오차(MSE). 값이 작을수록 복원력이 높음. |
| **(2) 분산 유지율 (Explained Variance Ratio)** | ![](https://latex.codecogs.com/svg.image?EVR_k%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Clambda_i%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Clambda_i%7D) | 상위 \(k\)개의 고유값이 전체 분산에서 차지하는 비율. PCA 등에서 정보 손실 정도 평가. |
| **(3) 상호 정보량 (Mutual Information)** | ![](https://latex.codecogs.com/svg.image?MI(X%2CY)%20%3D%20%5Csum_%7Bx%20%5Cin%20X%7D%20%5Csum_%7By%20%5Cin%20Y%7D%20p(x%2Cy)%5Clog%5Cfrac%7Bp(x%2Cy)%7D%7Bp(x)p(y)%7D) | 축소 전후 데이터의 정보량 비교. 값이 클수록 정보 보존이 잘됨. |
| **(4) 근접도 보존 – Trustworthiness** | ![](https://latex.codecogs.com/svg.image?T(k)%20%3D%201%20-%20%5Cfrac%7B2%7D%7Bnk(2n-3k-1)%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%20%5Cin%20U_k(i)%7D%20(r(i%2Cj)%20-%20k)) | 고차원에서 이웃이 아니던 점이 저차원에서 잘못 가까워지는 정도를 측정. |
| **(5) 근접도 보존 – Continuity** | ![](https://latex.codecogs.com/svg.image?C(k)%20%3D%201%20-%20%5Cfrac%7B2%7D%7Bnk(2n-3k-1)%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%20%5Cin%20V_k(i)%7D%20(r'(i%2Cj)%20-%20k)) | 저차원에서 이웃이던 점이 고차원에서 멀어지는 정도를 측정. |
| **(6) 거리 보존 – Stress (Kruskal’s Stress)** | ![](https://latex.codecogs.com/svg.image?Stress%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3Cj%7D(d_%7Bij%7D-%5Chat%7Bd%7D_%7Bij%7D)%5E2%7D%7B%5Csum_%7Bi%3Cj%7Dd_%7Bij%7D%5E2%7D%7D) | 고차원 거리와 저차원 거리 간의 차이 비율. 작을수록 거리 보존이 잘됨. |
| **(7) Sammon Error** | ![](https://latex.codecogs.com/svg.image?E_%7BSammon%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csum_%7Bi%3Cj%7D%20d_%7Bij%7D%7D%20%5Csum_%7Bi%3Cj%7D%20%5Cfrac%7B(d_%7Bij%7D-%5Chat%7Bd%7D_%7Bij%7D)%5E2%7D%7Bd_%7Bij%7D%7D) | 근접 관계를 강조한 거리 보존 오차. |
| **(8) LCMC (Local Continuity Meta Criterion)** | ![](https://latex.codecogs.com/svg.image?LCMC%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CN_H(i)%5Ccap%20N_L(i)%7C%7D%7Bk%7D%20-%20%5Cfrac%7Bk%7D%7Bn-1%7D) | 고차원/저차원 k-이웃의 겹침 비율로 지역/전역 구조를 함께 평가. |
| **(9) 쌍의 상관계수 (Spearman’s ρ)** | ![](https://latex.codecogs.com/svg.image?%5Crho%20%3D%201%20-%20%5Cfrac%7B6%5Csum_%7Bi%3D1%7D%5E%7BN%7D(r_i%20-%20s_i)%5E2%7D%7BN(N%5E2%20-%201)%7D) | 거리 순위 일관성을 평가. ρ=1이면 완전히 동일한 순서. |
| **(10) Silhouette Score** | ![](https://latex.codecogs.com/svg.image?s(i)%20%3D%20%5Cfrac%7Bb(i)%20-%20a(i)%7D%7B%5Cmax(a(i)%2C%20b(i))%7D) | 군집 간 거리 대비 군집 내 밀집도 평가. |
| **(11) Davies–Bouldin Index (DBI)** | ![](https://latex.codecogs.com/svg.image?DBI%20%3D%20%5Cfrac%7B1%7D%7Bk%7D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Cmax_%7Bj%5Cne%20i%7D%20%5Cfrac%7B%5Csigma_i%2B%5Csigma_j%7D%7Bd(c_i%2Cc_j)%7D) | 군집 내 분산과 군집 간 중심 거리의 비율. 낮을수록 좋음. |
| **(12) Adjusted Rand Index (ARI)** | ![](https://latex.codecogs.com/svg.image?ARI%20%3D%20%5Cfrac%7B%5Csum_%7Bij%7D%20%5Cbinom%7Bn_%7Bij%7D%7D%7B2%7D%20-%20%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%2F%5Cbinom%7Bn%7D%7B2%7D%7D%7B%5Cfrac%7B1%7D%7B2%7D%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%20%2B%20%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%20-%20%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%2F%5Cbinom%7Bn%7D%7B2%7D%7D) | 군집 일치도 평가. 1이면 완벽 일치, 0은 무작위 수준. |
| **(13) Normalized Mutual Information (NMI)** | ![](https://latex.codecogs.com/svg.image?NMI(U%2CV)%20%3D%20%5Cfrac%7B2I(U%3BV)%7D%7BH(U)%20%2B%20H(V)%7D) | 군집 결과와 실제 레이블 간의 상호 정보량을 정규화. 값이 1에 가까울수록 유사도가 높음. |

## [차원 축소 알고리즘 평가 사용방법]

**▣ 재구성 오류(Reconstruction Error) :** 차원 축소된 데이터를 원본 차원으로 복원하여 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)를 통해 재구성 오류를 계산<br>

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

    # (2) Davies-Bouldin Index (DBI) - 클러스터들이 얼마나 잘 분리되고 응집되어 있는지 평가(DBI가 낮을수록 클러스터링 품질이 더 좋음)
    davies_bouldin = davies_bouldin_score(X_reduced, y_pred)
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

    # (3) Adjusted Rand Index (ARI) - 실제 레이블과 예측 레이블 비교(클러스터링 결과와 실제 레이블 간의 일치도를 측정: 1에 가까울 수록 유사)
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    # (4) Normalized Mutual Information (NMI) - 실제 레이블과 예측 레이블 비교(클러스터링 결과와 실제 레이블 간의 정보량의 공유 정도를 측정: 1에 가까울 수록 유사)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

<br>

                        

