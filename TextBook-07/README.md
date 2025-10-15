#  07 : 군집 평가 지표

---
	
	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합

	  
---


| 지표명 | 의미 | 수식(대표 정의) |
|---|---|---|
| **[1] Silhouette Coefficient** | 한 점이 **자기 군집 평균거리** $a(i)$ 대비 **가장 가까운 다른 군집 평균거리** $b(i)$ 로 분리도 측정 | $s(i)=\dfrac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}$<br>$\bar{s}=\dfrac{1}{n}\sum_{i=1}^{n}s(i)$ |
| **[2] Davies–Bouldin Index (DBI)** | 군집 **내 응집도** 대비 **군집 간 분리도** (낮을수록 좋음) | $DBI=\dfrac{1}{K}\sum_{i=1}^{K}\max_{j\ne i}\dfrac{S_i+S_j}{M_{ij}}$<br>$S_i=\dfrac{1}{\lvert C_i\rvert}\sum_{x\in C_i}\lVert x-\mu_i\rVert,\; M_{ij}=\lVert\mu_i-\mu_j\rVert$ |
| **[3] Dunn Index (DI)** | **가장 가까운 군집 간 최소거리** 대비 **최대 군집 지름** (클수록 좋음) | $DI=\dfrac{\min_{i\ne j}\,\delta(C_i,C_j)}{\max_k\,\Delta(C_k)}$<br>$\delta$: 군집 간 최소거리, $\Delta$: 군집 지름(내부 최대거리) |
| **[4] Calinski–Harabasz Index (CHI)** | **군집 사이 분산 / 군집 내 분산** (클수록 좋음) | $CH=\dfrac{\mathrm{Tr}(B_K)/(K-1)}{\mathrm{Tr}(W_K)/(n-K)}$<br>$\mathrm{Tr}(B_K)=\sum_k \lvert C_k\rvert\,\lVert\mu_k-\bar{x}\rVert^2,\;\mathrm{Tr}(W_K)=\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$ |
| **[5] Within-Cluster Sum of Squares (WCSS)** | 각 점이 군집 **중심까지의 제곱거리 합** (작을수록 응집↑). K-means **목적함수와 동일** | $\mathrm{WCSS}=\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k\rVert^{2}$ |




---



