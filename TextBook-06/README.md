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
| 간단하고 이해하기 쉬움  | 모델 미생성으로 특징과 클래스 간 관계 이해가 제한적 |
| 학습 데이터분포 고려 불요 | 적절한 K의 선택이 필요 |
| 빠른 훈련 단계 | 데이터가 많아지면 느림 : 차원의 저주(curse of dimensionality) |
| 수치기반 데이터 분류 성능우수 | 명목특징 및 누락데이터위한 추가처리 필요(이상치에 민감)|

K-NN 모델은 각 변수들의 범위를 재조정(표준화, 정규화)하여 거리함수의 영향을 줄여야 한다.<br>
(1) 최소-최대 정규화(min-max normalization) : 변수 X의 범위를 0%에서 100%까지로 나타냄<br><br>
$X_{new} = \frac{X-min(X)}{max(X)-min(X)}$<br>

(2) z-점수 표준화(z-score standardization) : 변수 X의 범위를 평균의 위또는 아래로 표준편차만큼 떨어져 있는 지점으로 확대 또는 축소<br><br>
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
	# callable : 사용자가 정의한 함수(거리가 저장된 배열을 입력받고 가중치가 저장된 배열을 반환)
 	
	# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'} 
	# 가장 가까운 이웃들을 계산하는 데 사용하는 알고리즘을 결정한다. default는 auto이다. 
	# 'auto' : 입력된 훈련 데이터에 기반하여 가장 적절한 알고리즘을 사용한다. 
	# 'ball_tree' : Ball-Tree 구조를 사용한다.(https://nobilitycat.tistory.com/entry/ball-tree)
	# 'kd_tree' : KD-Tree 구조를 사용한다.
	# 'brute' : Brute-Force 탐색을 사용한다.  	
 	
	# leaf_size : int
	# Ball-Tree나 KD-Tree의 leaf size를 결정한다. default값은 30이다.
	# 트리를 저장하기 위한 메모리뿐만 아니라, 트리의 구성과 쿼리 처리의 속도에도 영향을 미친다. 
 	
	# p : int
	# 민코프스키 미터법(Minkowski)의 차수를 결정한다. 
	# p = 1이면 맨해튼 거리(Manhatten distance)
	# p = 2이면 유클리드 거리(Euclidean distance)이다. 

<br>

## k-최근접 이웃 분류(k-Nearest Neighbors Classification)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

	from sklearn.neighbors import KNeighborsClassifier
	kn = KNeighborsClassifier()

	#훈련
	kn.fit(train_input, train_target)
	#평가
	print(kn.score(test_input, test_target))

<br>
 
# [2] 서포트 벡터 머신 (Support Vector Machine, SVM)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/svm/index.html<br>
▣ 정의 : SVM은 N차원 공간을 (N-1)차원으로 나눌 수 있는 초평면을 찾는 분류 기법으로 2개의 클래스를 분류할 수 있는 최적의 경계를 찾는다.<br>

![](./images/margin.png)

- 최적의 경계 : 각 클래스의 말단에 위치한 데이터들 사이의 거리를 최대화 할 수 있는 경계<br>
- 초평면(hyper plane) : 고차원(N차원)에서 데이터를 두 분류로 나누는 결정 경계<br>
- Support Vector : 데이터들 중에서 결정 경계에 가장 가까운 데이터들<br>
- 마진(Margin) : 결정 경계와 support vector사이의 거리<br>
- 비용(Cost) : 마진(Margin) 크기의 반비례<br>
- 감마(Gamma) : train data 하나 당 결정 경계에 영향을 끼치는 범위를 조절하는 변수(크면 오버피팅, 작으면 언더피팅)<br>
- 커널 트릭(Kernel trick) : 비선형 분류를 하기 위해 차원을 높여줄 때마다 필요한 엄청난 계산량을 줄이기 위해 사용하는 커널 트릭은 실제로는 데이터의 특성을 확장하지 않으면서 특성을 확장한 것과 동일한 효과를 가져오는 기법<br>

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 과적합을 피할 수 있다 | 커널함수 선택이 명확하지 않다 |
| 분류 성능이 좋다 | 파라미터 조절을 적절히 수행하여 최적의 모델을 찾을 수 있다 |
| 저차원, 고차원 공간의 적은 데이터에 대해서 일반화 능력이 우수 | 계산량 부담이 있다 |
| 잡음에 강하다 | 데이터 특성의 스케일링에 민감하다|
| 데이터 특성이 적어도 좋은 성능 | | 

▣ 유형 : 선형SVM(하드마진, 소프트마진), 비선형SVM<br>
- 하드마진 : 두 클래스를 분류할 수 있는 최대마진의 초평면을 찾는 방법으로, 모든 훈련데이터는 마진의 바깥족에 위치하게 선형으로 구분해서 하나의 오차도 허용하면 안된다. 모든 데이터를 선형으로 오차없이 나눌 수 있는 결정경계를 찾는 것은 사실상 어렵다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|w\right\|^2$<br>

![](./images/hmargin.png)

- 소프트마진 :  하드마진이 가진 한계를 개선하고자 나온 개념으로, 완벽하게 분류하는 초평면을 찾는 것이 아니라 어느 정도의 오분류를 허용하는 방식이다. 소프트마진에서는 오분류를 허용하고 이를 고려하기 위해 slack variable을 사용하여 해당 결정경계로부터 잘못 분류된 데이터의 거리를 측정한다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|w\right\|^2 + C\sum_{i=1}^{n}\varepsilon i$

![](./images/smargin.png)

- 비선형분류 : 선형분리가 불가능한 입력공간을 선형분리가 가능한 고차원 특성공간으로 보내 선형분리를 진행하고 그 후 다시 기존의 입력공간으로 변환하면 비선형 분리를 하게 된다.<br><br>
입력공간을 특성공간으로 변환하기 위해서 mapping function을 사용한다<br>
$\Phi(x) = Ax$<br><br>
고차원의 특성공간으로 변환하고 목적함수에 대한 문제를 푸는 것이 간단한 차원에서는 가능하나 그 차수가 커질수록 계산량의 증가하는 것을 다시 해결하고자 나오는 개념이 커널트릭이다.<br>
$k(x_i, x_j) =\Phi(x_i)^T\Phi(x_j)$<br><br>
확장된 특성공간의 두 벡터의 내적만을 계산하여 고차원의 복잡한 계산 없이 커널 함수를 사용하여 연산량을 간단하게 해결할 수 있다. 가장 성능이 좋고 많이 사용되는 것이 가우시안 RBF(Radial basis function)이다.<br><br>
$k(x,y) = e^{-\frac{-\left\|x_i-x_j\right\|^2}{2\sigma^2}}$<br><br>
직접 차수를 정하는 방식(Polynomial) : $k(x,y) = (1+x^Ty)^p$<br>
신경망 학습(Signomail) : $k(x,y) = tanh(kx_ix_j-\delta)$<br>

<br>  

## 서포트 벡터 회귀 (Support Vector Regression, SVR)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html<br>
▣ 회귀식 : https://scikit-learn.org/stable/modules/svm.html#svr<br>

	from sklearn.svm import SVR
 
 	svr = SVR(kernel='rbf', gamma='auto')
	svr.fit(xtrain, ytrain)

	score = svr.score(xtest, ytest)
	print("R-squared: ", score)

<br> 

## 서포트 벡터 분류 (Support Vector Classification, SVC)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC<br>
▣ 회귀식 : https://scikit-learn.org/stable/modules/svm.html#svc<br>

	import sklearn.svm as svm

 	# 선형일 경우
	svm_clf =svm.SVC(kernel = 'linear')
 	# 비선형일 경우
 	svm_clf =svm.SVC(kernel = 'rbf')

	# 교차검증
	scores = cross_val_score(svm_clf, X, y, cv = 5)
 	scores.mean()

<br>

# [3] 결정 트리 (Decision Tree)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/index.html<br>

![](./images/tree.png)

▣ 정의 : 결정트리는 어떤 항목에 대한 관측값과 목표값을 연결시켜주는 예측 모델로, 대표적인 지도학습 분류 모델이며, 스무고개와 같이 질문에 대하여 '예' 또는 '아니오'를 결정하여 트리 구조를 나타낸다. 결정트리의 기본적 아이디어는 복잡도 감소시키는 것에 있다. 정보의 복잡도를 불순도(Impurity)라고 하며, 불순도를 수치화한 값에는 지니계수(Gini coefficient)와 엔트로피(Entropy)가 있다.<br><br>
 - 지니계수 : $G_i = 1-\sum_{k=1}^{n}P^2_{i,k}$<br>
 - 엔트로피 : $E_i = -\sum_{k=1}^{n}P_{i,k}log_2P_{i,k}$<br>
 
▣ 유형 :  ID3, CART
 - ID3 : 모든 독립변수가 범주형 데이터인 경우에만 분류가 가능하다. 정보획득량(Infomation Gain)이 높은 특징부터 분기해나가는데 정보획득량은 분기전 엔트로피와 분기후 엔트로피의 차이를 말한다.(엔트로피 사용)<br><br>
$IG(S, A) = E(S) - E(S|A)$<br>
 - CART : Classification and Regression Tree의 약자로, 이름 그대로 분류와 회귀가 모두 가능한 결정트리 알고리즘으로 yes 또는 no 두 가지로 분기한다.(지니계수 사용)<br><br> 
$f(k,t_k) = \frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}$<br>

<br>

# 결정 트리 회귀 (Decision Tree Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html<br>

<br>

# 결정 트리 분류 (Decision Tree Classification)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html<br>

<br>

# [4] 랜덤 포레스트 (Random Forest)  
▣ 가이드 : https://scikit-learn.org/stable/modules/ensemble.html#random-forests<br>

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 모델이 단순하며, 과적합이 잘 일어나지 않음 | 여러개의 결정트리 사용으로 메모리 사용량 큼 |
| 새로운 데이터에 일반화가 용이함 | 고차원 데이터나 희소 데이터에 잘 작동하지 않음 |

# 랜덤 포레스트 회귀 (Random Forest Regression)  
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#randomforestregressor<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor<br>

<br>

# 랜덤 포레스트 분류 (Random Forest Classification)    	  	
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#randomforestclassifier<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier<br>

<br>



