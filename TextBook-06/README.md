#  06 : 지도 학습 (Supervised Learning, SL) : 회귀 (regression) + 분류 (classification)

---

	[1] k-최근접 이웃 (k-Nearest Neighbors, K-NN) 	
		k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
		k-최근접 이웃 분류(k-Nearest Neighbors Classification)
 	[2] 서포트 벡터 머신 (Support Vector Machine, SVM)
		서포트 벡터 회귀 (Support Vector Regression, SVR)
		서포트 벡터 분류 (Support Vector Classification, SVC)
	[3] 결정 트리 (Decision Tree)
 		결정 트리 회귀 (Decision Tree Regression)
   		결정 트리 분류 (Decision Tree Classification)
 	[4] 랜덤 포레스트 (Random Forest)  
		랜덤 포레스트 회귀 (Random Forest Regression)  
		랜덤 포레스트 분류 (Random Forest Classification)    	  	
	
---  

# [1] k-최근접 이웃 (k-Nearest Neighbors, K-NN) 	
▣ 가이드 : https://scikit-learn.org/stable/modules/neighbors.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/neighbors/index.html<br>

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 단순하고 효율적                  | 모델 미생성으로 특징과 클래스 간 관계 이해 제한적 |
| 기저 데이터분포에 대한 가정 불요 | 적절한 K의 선택이 필요                            |
| 빠른 훈련 단계                   | 데이터가 많아지면 분류단계가 느리다               |
| 수치기반 데이터 분류 성능 우수   | 명목특징 및 누락데이터위한 추가처리 필요          |

K-NN 모델은 각 변수들의 범위를 재조정(표준화, 정규화)하여 거리함수의 영향을 줄여야 한다.<br>
(1) 최소-최대 정규화(min-max normalization) : 변수 X의 범위를 0%에서 100%까지로 나타냄<br>
$X_{new} = \frac{X-min(X)}{max(X)-min(X)}$<br>

(2) z-점수 표준화(z-score standardization) : 변수 X의 범위를 평균의 위또는 아래로 표준편차만큼 떨어져 있는 지점으로 확대 또는 축소<br>
$X_{new} = \frac{X-\mu}{\sigma}= \frac{X-min(X)}{StdDev(X)}$

<br>

## k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html<br>
주변의 가장 가까운 K개의 샘플 평균을 통해 값을 예측하는 방식이다.<br> 
한계 : 테스트하고자 하는 샘플에 근접한 훈련 데이터가 없는 경우, 즉 훈련 셋의 범위를 많이 벗어나는 샘플인 경우 정확하게 예측하기 어렵다. 

	class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', 
	leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
	 
	# n_neighbors : int
	# 이웃의 수인 K를 결정한다. default는 5다. 
	 
  	# weights : {'uniform', 'distance'} or callable
	# 예측에 사용되는 가중 방법을 결정한다. default는 uniform이다. 
	# 'uniform' : 각각의 이웃이 모두 동일한 가중치를 갖는다. 
	# 'distance' : 거리가 가까울수록 더 높은 가중치를 가져 더 큰 영향을 미치게 된다.
	# callable : 사용자가 직접 정의한 함수를 사용할 수도 있다. 거리가 저장된 배열을 입력받고 가중치가 저장된 배열을 반환하는 함수
 	
	# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'} 
	# 가장 가까운 이웃들을 계산하는 데 사용하는 알고리즘을 결정한다. default는 auto이다. 
	# 'auto' : 입력된 훈련 데이터에 기반하여 가장 적절한 알고리즘을 사용한다. 
	# 'ball_tree' : Ball-Tree 구조를 사용한다.(Ball-Tree 설명 : https://nobilitycat.tistory.com/entry/ball-tree)
	# 'kd_tree' : KD-Tree 구조를 사용한다.
	# 'brute' : Brute-Force 탐색을 사용한다.  	
 	
	# leaf_size : int
	# Ball-Tree나 KD-Tree의 leaf size를 결정한다. default값은 30이다.
	# 이는 트리를 저장하기 위한 메모리뿐만 아니라, 트리의 구성과 쿼리 처리의 속도에도 영향을 미친다. 
 	
	# p : int
	# 민코프스키 미터법(Minkowski)의 차수를 결정한다. 
	# 예를 들어 p = 1이면 맨해튼 거리(Manhatten distance), 
	# p = 2이면 유클리드 거리(Euclidean distance)이다. 

<br>

## k-최근접 이웃 분류(k-Nearest Neighbors Classification)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

<br>
 
# [2] 서포트 벡터 머신 (Support Vector Machine, SVM)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/svm/index.html<br>
분류를 위한 선형 혹은 비선형 결정 경계(Decision Boundary)를 정하는 모델
선형 분류에서 명확하게 결정경계를 그을 수 없는 상황에서는 데이터의 특징을 추가함으로써 차원을 늘려서 분류가 필요
고차원(N차원)에서 데이터를 두 분류로 나누는 결정 경계를 초평면(hyper plane)이라고 한다.
선형 분류로 구분할 수 없는 데이터를 비선형 분류까지 이용하여 분류하는 분류기를 SVM이라고 한다.
비선형 분류를 하기 위해 차원을 높여줄 때마다 필요한 엄청난 계산량을 줄이기 위해 SVM 구조에서 커널 트릭(Kernel trick)을 사용
커널 트릭은 실제로는 데이터의 특성을 확장하지 않으면서 특성을 확장한 것과 동일한 효과를 가져오는 기법이다.
SVM에서 Support Vector는 데이터들 중에서 결정 경계에 가장 가까운 데이터들을 의미하며, 이때 결정 경계와 support vector사이의 거리를 마진(Margin)이라고 한다.
이 마진을 이용하여 최적의 결정 경계를 찾아내는데, 각 데이터 그룹의 support vector의 마진이 가장 크게(결정 경계의 쏠림 방지) 결정 경계를 잡아야한다.
Cost : 마진(Margin) 크기의 반비례
Gamma : train data 하나 당 결정 경계에 영향을 끼치는 범위를 조절하는 변수(크면 오버피팅, 작으면 언더피팅)



## 서포트 벡터 회귀 (Support Vector Regression, SVR)
https://scikit-learn.org/stable/auto_examples/svm/index.html

## 서포트 벡터 분류 (Support Vector Classification, SVC)

<br>

# [3] 결정 트리 (Decision Tree)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/index.html<br>

# 결정 트리 회귀 (Decision Tree Regression)
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html<br>

# 결정 트리 분류 (Decision Tree Classification)

<br>

# [4] 랜덤 포레스트 (Random Forest)  

# 랜덤 포레스트 회귀 (Random Forest Regression)  
# 랜덤 포레스트 분류 (Random Forest Classification)    	  	

<br>



