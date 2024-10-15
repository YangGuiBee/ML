#  10 : 비지도 학습 (Unsupervised Learning, UL) : 군집화

---

	[1] KDE (Kernel Desity Estimation)
 	[2] k-평균 클러스터링 (k-Means Clustering)
	[3] 계층적 클러스터링 (Hierarchical Clustering)
	[4] DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
	[5] 가우시안 혼합 모델 (Gaussian Mixture Model, GMM)
   
---  

# [1] KDE (Kernel Desity Estimation)

# [2] k-평균 클러스터링 (k-Means Clustering)
▣ 정의 : 데이터를 K개의 군집으로 나누고 각 군집의 중심(centroid)을 기준으로 데이터를 반복적으로 할당하는 군집화 알고리즘<br>
▣ 필요성 : 데이터를 그룹화하여 숨겨진 패턴을 발견하는 데 유용<br>
▣ 장점 : 구현이 간단하고 계산 속도가 빠르며, 대규모 데이터셋에 적합<br>
▣ 단점 : 군집의 개수(K)를 사전에 정의해야 하며, 구형 군집이 아니거나 이상치(outliers)가 있을 경우 성능 저하<br>
▣ 응용분야 : 고객 세분화, 이미지 분할, 추천 시스템<br>
▣ 모델식 : 𝐾는 군집의 개수, $𝐶_𝑖$는 i번째 군집, $𝜇_𝑖$는 i번째 군집의 중심, 𝑥는 데이터 포인트<br>

	from sklearn.cluster import KMeans
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	kmeans = KMeans(n_clusters=3, random_state=0)
	kmeans.fit(X)
	labels = kmeans.labels_

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
	plt.title("K-Means Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

# [3] 계층적 클러스터링 (Hierarchical Clustering)

# [4] DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

# [5] 가우시안 혼합 모델 (Gaussian Mixture Model, GMM)


