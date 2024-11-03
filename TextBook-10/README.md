#  10 : 비지도 학습(Unsupervised Learning, UL) : 군집화

---

	[1] k-평균 클러스터링(k-Means Clustering)
	[2] 계층적 클러스터링(Hierarchical Clustering)
	[3] 밀도기반 군집화(Density-Based Spatial Clustering of Applications with Noise, DBSCAN)
	[4] 가우시안 혼합 모델(Gaussian Mixture Model, GMM)
	[5] 평균이동 군집화(Mean-Shift Clustering)
	[6] 스펙트럼 군집화(Spectral Clustering)
	[7] 군집구조 식별을 위한 점 순서화(Ordering Points To Identify the Clustering Structure, OPTICS)
	[8] 친화도 전파(Affinity Propagation)
   
---  

## 클러스터링(k-Means Clustering)이란?
데이터 포인트들을 별개의 군집으로 그룹화하는 것<br>
유사성이 높은 데이터들을 동일한 그룹으로 분류하고 서로다른 군집들이 상이성을 가지도록 그룹화<br>
군집화 활용분야 : 고객, 시장, 상품, 경제 및 사회활동 등의 세분화(Segmentation) → 이미지 식별, 이상검출 등<br>

<br>

# [1] k-평균 클러스터링(k-Means Clustering)
▣ 정의 : 데이터를 K개의 군집으로 나누고 각 군집의 중심점(centroid)을 기준으로 데이터를 반복적으로 할당하는 군집화 알고리즘<br>

	k-Means Clustering에서 최초의 군집중심점(centroid) 선정방법

	① 무작위 선택(Random Initialization) : 데이터 포인트 중에서 k개의 점을 무작위로 선택하여 초기 중심점으로 사용(간단하지만 결과가 일관되지 않을 가능성)
	② k-means++ 초기화 : 첫 번째 중심점을 무작위로 선택한 후, 나머지 중심점은 현재 선택된 중심점에서의 거리가 가장 먼 점을 선택(군집의 분산을 줄이고 수렴 속도를 개선)
	③ 특정 기준에 따라 선택 : 데이터의 분포나 도메인 지식을 바탕으로 특정 기준을 사용하여 초기 중심점을 선정
 
▣ 필요성 : 데이터를 그룹화하여 숨겨진 패턴을 발견하는 데 유용<br>
▣ 장점 : 구현이 간단하고 계산 속도가 빠르며, 대규모 데이터셋에 적합<br>
▣ 단점 : 군집의 개수(K)를 사전에 정의해야 하며, 구형 군집이 아니거나 이상치(outliers)가 있을 경우 성능 저하<br>
▣ 응용분야 : 고객 세분화, 이미지 분할, 추천 시스템<br>
▣ 모델식 : 𝐾는 군집의 개수, $𝐶_𝑖$는 i번째 군집, $𝜇_𝑖$는 i번째 군집의 중심, 𝑥는 데이터 포인트<br>

J=\sum_{i=1}^{K}\sum_{x\in C_i}^{}\left\| x-\mu_i \right\|^2<br>

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

<br>

# [2] 계층적 클러스터링(Hierarchical Clustering)
▣ 정의 : 데이터를 병합(bottom-up)하거나 분할(top-down)하여 계층적인 군집 구조를 만드는 방법<br>
▣ 필요성 : 군집의 개수를 사전에 정할 필요 없이 계층적 관계를 파악할 때 사용<br>
▣ 장점 : 군집 수를 미리 정할 필요 없으며, 덴드로그램(dendrogram)을 통한 군집 분석 가능<br>
▣ 단점 : 계산 복잡도가 높으며, 초기 병합 또는 분할 결정이 최종 결과에 영향을 줄 수 있음<br>
▣ 응용분야 : 계통수 분석, 텍스트 및 문서 분류<br> 
▣ 모델식 : $𝐶_𝑖$와 $𝐶_𝑗$는 각각 두 군집이고, 𝑑(𝑥,𝑦)는 두 데이터 포인트 𝑥와 𝑦 간의 거리<br>

	from scipy.cluster.hierarchy import dendrogram, linkage
	import matplotlib.pyplot as plt
	from sklearn.datasets import load_iris

	iris = load_iris()
	X = iris.data

	Z = linkage(X, 'ward')  # ward: 최소분산 기준 병합

	plt.figure(figsize=(10, 5))
	dendrogram(Z)
	plt.title("Hierarchical Clustering Dendrogram")
	plt.xlabel("Sample Index")
	plt.ylabel("Distance")
	plt.show()

▣ 덴드로그램(dendrogram) : 나무(tree) 모양의 도식으로, 계층적 군집화의 결과를 시각화하는 데 사용된다. 이 그래프는 각 데이터 포인트가 병합되거나 분할되는 과정을 계층 구조로 표현하며, 군집 간의 관계를 직관적으로 이해할 수 있도록 도와준다. 덴드로그램의 구조는 다음과 같다:<br>
(1) 각 데이터 포인트는 맨 아래에서 개별 노드로 시작 : 덴드로그램에서 각 데이터 포인트는 맨 아래에 위치한 개별 노드로 시작합니다. 이 단계에서는 각각의 데이터가 하나의 군집을 이루고 있다.<br>
(2) 데이터 포인트들이 병합 : 계층적 군집화의 과정에서 유사한 데이터 포인트끼리 순차적으로 병합되며, 병합되는 과정이 덴드로그램에서 상위로 올라가면서 두 노드가 연결되는 형태로 시각화 된다.<br>
(3) 병합된 군집이 다시 다른 군집과 병합 : 유사한 군집끼리 계속 병합되며 점점 더 큰 군집을 형성하게 된다. 덴드로그램의 상단으로 갈수록 더 큰 군집이 병합된 결과를 나타내며, 결국 모든 데이터가 하나의 군집으로 병합된다.<br>
(4) 군집 간의 거리 정보: 덴드로그램에서 두 군집이 병합된 높이(수직 축)는 그 두 군집 사이의 유사도 또는 거리를 나타낸다. 즉, 병합된 높이가 클수록 두 군집 간의 거리가 더 멀었다는 것을 의미합니다. 이는 데이터를 나누거나 군집을 형성하는 데 있어 중요한 기준이 된다.<br>
덴드로그램의 장점은 다음과 같다:<br>
(1) 군집의 개수 선택이 유연 : 덴드로그램을 통해 데이터가 어떻게 군집화되었는지 시각적으로 확인한 후, 임의의 높이에서 선을 그어 군집의 개수를 선택할 수 있다. 특정 높이에서 덴드로그램을 자르면 그 높이 기준으로 몇 개의 군집이 형성되는지를 알 수 있으며 이로 인해 군집의 개수를 미리 결정하지 않고도 군집을 형성할 수 있다. 예를 들어, 덴드로그램에서 각 군집 간의 유사도가 높지 않다고 판단되는 지점에서 잘라내면 다수의 작은 군집이 만들어질 수 있고, 반대로 유사도가 높다고 판단되는 지점에서 자르면 소수의 큰 군집이 형성될 수 있다.<br>
(2) 군집 간의 유사도 및 계층 구조 파악 : 덴드로그램은 단순히 군집을 나누는 것 이상으로 군집 간의 유사도와 계층적 관계를 직관적으로 보여준다. 이를 통해 두 군집이 병합되는 시점과 그 군집들이 다른 군집들과 얼마나 유사한지를 파악하고 이 정보를 바탕으로 군집화 결과를 더욱 상세하게 해석할 수 있다.<br>
(3) 다양한 수준에서 군집 분석 가능 : 덴드로그램을 활용하면 데이터셋을 다양한 수준에서 분석할 수 있다. 특정 높이에서 군집을 잘라내면 더 큰 군집을 형성할 수 있고, 더 낮은 높이에서는 세부적인 군집을 식별함으로써 다단계 군집 분석을 가능하게 한다.<br>
(4) 군집의 구조적 관계 시각화 : 덴드로그램을 통해 데이터를 계층적으로 군집화한 결과를 시각적으로 확인함으로써 데이터가 점진적으로 어떻게 병합되는지, 그리고 군집화가 특정 기준에 따라 어떻게 변하는지를 쉽게 이해할 수 있다.<br>

	from scipy.cluster.hierarchy import dendrogram, linkage
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	# 데이터 로드
	iris = load_iris()
	X = iris.data

	# 계층적 군집화 수행
	Z = linkage(X, 'ward')  # ward: 최소분산 기준 병합

	# 덴드로그램 시각화
	plt.figure(figsize=(10, 5))
	dendrogram(Z)
	plt.title("Hierarchical Clustering Dendrogram")
	plt.xlabel("Sample Index")
	plt.ylabel("Distance")
	plt.show()

<br>

# [3] 밀도기반 군집화(Density-Based Spatial Clustering of Applications with Noise, DBSCAN)
▣ 정의 : 밀도가 높은 영역을 군집으로 묶고, 밀도가 낮은 점들은 노이즈로 간주하는 밀도 기반 군집화 알고리즘<br>
▣ 필요성 : 다양한 밀도의 데이터 군집화 및 이상치 탐지에 유용<br>
▣ 장점 : 군집의 개수를 사전 설정할 필요 없으며, 이상치(outliers)를 자연스럽게 처리 가능<br>
▣ 단점 : 적절한 파라미터(ε, MinPts) 설정이 필요하며, 밀도가 균일하지 않은 데이터에 부적합<br>
▣ 응용분야 : 이상 탐지, 지리적 데이터 분석<br>
▣ 모델식: 각 점에서 반경 𝜖 내에 있는 점들이 미리 정의된 MinPts 보다 많으면 그 점을 중심으로 군집을 형성<br>

	from sklearn.cluster import DBSCAN
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	dbscan = DBSCAN(eps=0.5, min_samples=5)
	labels = dbscan.fit_predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
	plt.title("DBSCAN Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>

# [4] 가우시안 혼합 모델(Gaussian Mixture Model, GMM)
▣ 정의 : 여러 가우시안 분포(Gaussian Distribution)를 사용해 데이터를 모델링하고, 각 데이터 포인트가 각 분포에 속할 확률을 계산하는 군집화 방법<br>
▣ 필요성 : 복잡한 데이터 분포를 유연하게 모델링하여 군집 경계를 확률적으로 표현할 수 있음<br>
▣ 장점 : 데이터가 여러 분포를 따를 때 적합하며, 군집 간의 경계가 확률적으로 처리<br>
▣ 단점 : 초기화에 민감하고 계산 비용이 높음<br>
▣ 응용분야 : 패턴 인식, 이미지 세분화<br>
▣ 모델식 : $π_k$는 가우시안의 가중치, $𝜇_𝑘$, $Σ_𝑘$는 각각 평균과 공분산<br>

	from sklearn.mixture import GaussianMixture
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	gmm = GaussianMixture(n_components=3)
	gmm.fit(X)
	labels = gmm.predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
	plt.title("GMM Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>

# [5] 평균이동 군집화(Mean-Shift Clustering)
▣ 정의 : 데이터의 밀도가 높은 방향으로 이동하며 군집의 중심을 찾는 비모수 군집화 방법<br>
▣ 필요성 : 군집의 개수를 사전 설정할 필요 없이 자연스러운 군집을 찾을 때 유용<br>
▣ 장점 : 군집 개수 사전 설정 불필요하며, 비선형적 분포에도 적합<br>
▣ 단점 : 계산 비용이 크고 고차원 데이터에 적합하지 않음<br>
▣ 응용분야 : 이미지 세그멘테이션, 객체 추적<br>
▣ 모델식 : 𝐾는 커널 함수, 𝑥는 이동할 점, 𝑁(𝑥)는 반경 내 이웃 점<br>

	from sklearn.cluster import MeanShift
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	mean_shift = MeanShift()
	labels = mean_shift.fit_predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='cool')
	plt.title("Mean-Shift Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>

# [6] 스펙트럼 군집화(Spectral Clustering)
▣ 정의 : 그래프 이론을 기반으로 데이터의 유사도 행렬(Similarity Matrix)을 사용해 저차원 공간에서 군집을 찾는 알고리즘<br>
▣ 필요성 : 복잡한 구조를 가진 데이터에서 비선형적인 경계를 정의할 수 있는 군집화 방법이 필요할 때 유용<br>
▣ 장점 : 비선형적인 데이터에도 유용하며, 전통적인 군집화 알고리즘보다 복잡한 데이터 구조 처리 가능<br>
▣ 단점 : 유사도 행렬을 계산해야 하므로 메모리 사용량이 크고, 대규모 데이터에 비효율적<br>
▣ 응용분야 : 이미지 분할, 네트워크 분석<br>
▣ 모델식 : 𝐿은 라플라시안 행렬, 𝐷는 대각 행렬(각 노드의 차수), 𝐴는 인접 행렬입니다. 이 라플라시안 행렬의 고유벡터를 사용해 데이터를 군집화<br>
$𝐿=𝐷−𝐴$<br>

	from sklearn.cluster import SpectralClustering
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)
	labels = spectral.fit_predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm')
	plt.title("Spectral Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>

# [7] 군집구조 식별을 위한 점 순서화(Ordering Points To Identify the Clustering Structure, OPTICS)
▣ 정의 : 밀도 기반 군집화(DBSCAN)의 확장으로, 여러 밀도 수준에서 데이터의 군집 구조를 식별할 수 있는 알고리즘<br>
▣ 필요성 : 다양한 밀도를 가진 데이터에서 군집을 찾아내고 이상치(outliers)를 처리할 때 유용<br>
▣ 장점 : DBSCAN과 유사하게 이상치를 감지할 수 있으며, 여러 밀도 수준에서 군집을 식별 가능<br>
▣ 단점 : 계산 시간이 오래 걸릴 수 있으며, 적절한 매개변수 설정이 어려울 수 있음<br>
▣ 응용분야 : 지리적 데이터 분석, 이상치 탐지<br>
▣ 모델식 : DBSCAN과 유사하게 밀도 기반 접근을 따르며, 각 데이터 포인트의 reachability-distance와 core-distance를 기반으로 군집 구조를 형성<br>

	from sklearn.cluster import OPTICS
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	optics = OPTICS(min_samples=5)
	labels = optics.fit_predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
	plt.title("OPTICS Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>

# [8] 친화도 전파(Affinity Propagation)
▣ 정의 : 데이터 간의 유사도(similarity) 행렬을 사용해 가장 적합한 중심(exemplar)을 선택하여 군집을 형성하는 알고리즘<br>
▣ 필요성 : 군집의 개수를 미리 정할 필요 없이 데이터의 유사도에 기반해 자연스럽게 군집을 찾을 수 있음<br>
▣ 장점 : 군집 개수를 사전에 정의할 필요 없으며, 유사도에 기반한 군집화로 군집 경계가 더 명확할 수 있음<br>
▣ 단점 : 계산 비용이 크고 큰 데이터셋에서는 느릴 수 있음<br>
▣ 응용분야 : 이미지 분할, 문서 분류<br>
▣ 모델식: 각 데이터 포인트 간의 유사도 𝑠(𝑖,𝑘)와 책임 𝑟(𝑖,𝑘), 가용도 𝑎(𝑖,𝑘)를 반복적으로 계산해 중심점을 결정<br>

	from sklearn.cluster import AffinityPropagation
	from sklearn.datasets import load_iris
	import matplotlib.pyplot as plt

	iris = load_iris()
	X = iris.data

	affinity_propagation = AffinityPropagation(random_state=0)
	labels = affinity_propagation.fit_predict(X)

	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
	plt.title("Affinity Propagation Clustering on Iris Dataset")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

<br>
<br>
