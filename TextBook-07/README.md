#  07 : 군집 평가 지표

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
| **[3] Dunn Index (DI)**               | ✓ 높을수록 좋음 (상한 데이터 의존) | **≥ 0.50 매우 좋음**(드묾), **0.30~0.49 좋음**, **0.10~0.29 보통**, **< 0.10 낮음** | 최소 군집 간 거리 / 최대 지름이라 값이 전반적으로 작게 나오는 편. 이상치·밀도 차이에 민감. |
| **[4] Calinski–Harabasz Index (CHI)** | ✓ 높을수록 좋음             | **절대 임계치 없음** → 동일 데이터에서 k 간 상대 비교: 국소 최대/엘보우 부근이 바람직           | 데이터/스케일에 강하게 의존. 보통 k 스윕으로 비교.                             |
| **[5] Within-Cluster Sum of Squares (WCSS)**                | ✗ 낮을수록 좋음             | **절대 임계치 없음** → 엘보우(굽어지는 지점)에서 k 선택                                 | 표준화 여부·특징 개수에 좌우. k↑→단조감소이므로 상대 비교 전용.                 |



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

---



