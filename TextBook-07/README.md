#  07 : 군집 평가 지표

---
	
	[1] Silhouette Coefficient (실루엣 계수)
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Inertia (Within-Cluster Sum of Squares, WCSS)

	  
---


| 지표명                                 | 의미                                                                     | 수식(대표 정의)                                                                                                                                                                                                  | 적용 대상 모델                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Silhouette Coefficient (실루엣 계수)** | 한 점이 **자기 군집과의 평균거리** (a(i)) 대비 **가장 가까운 다른 군집과의 평균거리** (b(i))로 분리도 측정 | ( s(i)=\dfrac{b(i)-a(i)}{\max{a(i),,b(i)}} )<br>전체 점 평균: (\tfrac{1}{n}\sum_i s(i))                                                                                                                         | **거리 기반 전반** (K-means, K-medoids, 계층형, 스펙트럴, DBSCAN/OPTICS*)                                        |
| **Davies–Bouldin Index (DBI)**      | 군집 **내 응집도** 대비 **군집 간 분리도** (낮을수록 좋음)                                 | ( DBI=\dfrac{1}{K}\sum_{i=1}^{K}\max_{j\ne i}\dfrac{S_i+S_j}{M_{ij}} )<br>(S_i): 군집 (i) 평균 내거리, (M_{ij}=\lVert \mu_i-\mu_j \rVert)                                                                         | **거리 기반 전반**                                                                                        |
| **Dunn Index (DI)**                 | **가장 가까운 군집 간 최소거리** 대비 **최대 군집 지름** (클수록 좋음)                          | ( DI=\dfrac{\min_{i\ne j},\delta(C_i,C_j)}{\max_k,\Delta(C_k)} )<br>(\delta): 군집 간 최소거리, (\Delta): 군집 지름(내부 최대거리)                                                                                          | **거리 기반 전반**                                                                                        |
| **Calinski–Harabasz Index (CHI)**   | **군집 사이 분산 / 군집 내 분산** (클수록 좋음)                                        | ( CH=\dfrac{\mathrm{Tr}(B_K)/(K-1)}{\mathrm{Tr}(W_K)/(n-K)} )<br>(\mathrm{Tr}(B_K)=\sum_k \lvert C_k \rvert,\lVert \mu_k-\bar{x} \rVert^2,;\mathrm{Tr}(W_K)=\sum_k\sum_{x\in C_k}\lVert x-\mu_k \rVert^2 ) | **중심/거리 기반 전반** (K-means, K-medoids, **계층형**, **스펙트럴**, **GMM–하드할당**, 미니배치 K-means; **유클리드 거리 권장**) |
| **Inertia (WCSS)**                  | 각 점이 군집 **중심까지의 제곱거리 합** (작을수록 응집 ↑). K-means **목적함수와 동일**             | ( \mathrm{WCSS}=\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k \rVert^2 )                                                                                                                                     | **중심 기반** (K-means/미니배치 K-means). 비유클리드·밀도기반에는 부적합                                                  |



---



