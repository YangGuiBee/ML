#  06 : 지도 학습 (Supervised Learning, SL) : 회귀 (regression) + 분류 (classification)

---

	[1] k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
 	[2] 서포트 벡터 회귀 (Support Vector Regression, SVR)
	[3] 결정 트리 회귀 (Decision Tree Regression)
 	[4] 랜덤 포레스트 회귀 (Random Forest Regression)  
  	
  	[5] k-최근접 이웃 분류(k-Nearest Neighbors Classification)
  	[6] 서포트 벡터 분류 (Support Vector Classification, SVC)
   	[7] 결정 트리 분류 (Decision Tree Classification)
   	[8] 랜덤 포레스트 분류 (Random Forest Classification)    	
	
---  

# [1] k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html<br>
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

# [2] 서포트 벡터 회귀 (Support Vector Regression, SVR)

<br>

# [3] 결정 트리 회귀 (Decision Tree Regression)

<br>

# [4] 랜덤 포레스트 회귀 (Random Forest Regression)  

<br>

# [5] k-최근접 이웃 분류(k-Nearest Neighbors Classification)
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

<br>

# [6] 서포트 벡터 분류 (Support Vector Classification, SVC)

<br>

# [7] 결정 트리 분류 (Decision Tree Classification)

<br>

# [8] 랜덤 포레스트 분류 (Random Forest Classification)

<br>

