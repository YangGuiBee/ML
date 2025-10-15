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


## ▣ 평가지표 결과해석

| 지표                                | 목표                    | 권장 해석 기준(경험치)                                                           | 비고                                                             |
| --------------------------------- | --------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------- |
| **[1] Silhouette Coefficient**        | ✓ 높을수록 좋음 (범위 −1~1)   | **≥ 0.70 매우 좋음**, **0.50~0.69 양호**, **0.25~0.49 보통**, **< 0.25 미흡**     | 군집 내 응집 vs 인접 군집과 분리. 평균값과 군집별 분포를 함께 확인 권장.               |
| **[2] Davies–Bouldin Index (DBI)**    | ✗ 낮을수록 좋음 (하한 0)      | **≤ 0.50 매우 좋음**, **0.51~0.99 양호**, **1.00~1.49 보통**, **≥ 1.50 미흡**     | 군집 응집도 대비 중심 간 분리. 스케일·거리척도에 민감.                           |
| **[3] Dunn Index (DI)**               | ✓ 높을수록 좋음 (상한 데이터 의존) | **≥ 0.50 매우 좋음**(드묾), **0.30~0.49 좋음**, **0.10~0.29 보통**, **< 0.10 낮음** | 최소 군집 간 거리 / 최대 지름이라 값이 전반적으로 작게 나오는 편. 이상치·밀도 차이에 민감. |
| **[4] Calinski–Harabasz Index (CHI)** | ✓ 높을수록 좋음             | **절대 임계치 없음** → 동일 데이터에서 k 간 상대 비교: 국소 최대/엘보우 부근이 바람직           | 데이터/스케일에 강하게 의존. 보통 k 스윕으로 비교.                             |
| **[5] Within-Cluster Sum of Squares (WCSS)**                | ✗ 낮을수록 좋음             | **절대 임계치 없음** → 엘보우(굽어지는 지점)에서 k 선택                                 | 표준화 여부·특징 개수에 좌우. k↑→단조감소이므로 상대 비교 전용.                 |


---



