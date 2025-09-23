


#  11 : 지도 학습(Supervised Learning, SL) : 회귀(regression) + 분류(classification)

---

	[1] k-최근접 이웃(k-Nearest Neighbors, K-NN) 	
		k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
		k-최근접 이웃 분류(k-Nearest Neighbors Classification)
 	[2] 서포트 벡터 머신(Support Vector Machine, SVM)
		서포트 벡터 회귀(Support Vector Regression, SVR)
		서포트 벡터 분류(Support Vector Classification, SVC)
	[3] 결정 트리(Decision Tree)
 		결정 트리 회귀(Decision Tree Regression)
   		결정 트리 분류(Decision Tree Classification)
 	[4] 랜덤 포레스트(Random Forest) : 앙상블 학습(Ensemble Learning)에 해당(12강)
		랜덤 포레스트 회귀(Random Forest Regression)  
		랜덤 포레스트 분류(Random Forest Classification)    	  	
	
---  

# [1] k-최근접 이웃(k-Nearest Neighbors, K-NN) 	
▣ 가이드 : https://scikit-learn.org/stable/modules/neighbors.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/neighbors/index.html<br>
▣ 정의 : 머신러닝에서 데이터를 가장 가까운 유사속성에 따라 분류하여 데이터를 거리기반으로 분류분석하는 기법으로,<br>
비지도학습인 군집화(Clustering)과 유사한 개념이나 기존 관측치의 Y값이 존재한다는 점에서 지도학습에 해당한다.<br>

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 간단하고 이해하기 쉬움  | 모델 미생성으로 특징과 클래스 간 관계 이해가 제한적 |
| 학습 데이터분포 고려 불요 | 적절한 K의 선택이 필요 |
| 빠른 훈련 단계 | 데이터가 많아지면 느림 : 차원의 저주(curse of dimensionality) |
| 수치기반 데이터 분류 성능우수 | 명목특징 및 누락데이터위한 추가처리 필요(이상치에 민감)|

데이터로부터 거리가 가까운 'K'개의 다른 데이터의 레이블을 참조하여 분류할때 거리측정은 유클리디안 거리 계산법을 사용한다.<br>
![](./images/distance.PNG)

K-NN 모델은 각 변수들의 범위를 재조정(표준화, 정규화)하여 거리함수의 영향을 줄여야 한다.<br>
(1) 최소-최대 정규화(min-max normalization) : 변수 X의 범위를 0(0%)에서 1(100%)사이로 나타냄<br><br>
$X_{new} = \frac{X-min(X)}{max(X)-min(X)}$<br>

(2) z-점수 표준화(z-score standardization) : 변수 X의 범위를 평균의 위또는 아래로 표준편차만큼 떨어져 있는 지점으로 확대 또는 축소(데이터를 평균 0, 표준편차 1로 변환)하는 방식으로, 데이터의 중심을 0으로 맞추고, 데이터를 단위 표준 편차로 나누어 값을 재조정<br><br>
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
 
# [2] 서포트 벡터 머신(Support Vector Machine, SVM)
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


| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 과적합을 피할 수 있다 | 커널함수 선택이 명확하지 않다 |
| 분류 성능이 좋다 | 파라미터 조절을 적절히 수행해야만 최적의 모델을 찾을 수 있다 |
| 저차원, 고차원 공간의 적은 데이터에 대해서 일반화 능력이 우수 | 계산량 부담이 있다 |
| 잡음에 강하다 | 데이터 특성의 스케일링에 민감하다|
| 데이터 특성이 적어도 좋은 성능 | | 

▣ 유형 : 선형SVM(하드마진, 소프트마진), 비선형SVM<br>
- 하드마진 : 두 클래스를 분류할 수 있는 최대마진의 초평면을 찾는 방법으로, 모든 훈련데이터는 마진의 바깥족에 위치하게 선형으로 구분해서 하나의 오차도 허용하면 안된다. 모든 데이터를 선형으로 오차없이 나눌 수 있는 결정경계를 찾는 것은 사실상 어렵다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|\left\|w\right\|\right\|^2$     제약 조건은 모든 i에 대해 $𝑦_𝑖(𝑤⋅𝑥_𝑖+𝑏)≥1$ <br>

![](./images/hmargin.png)

- 소프트마진 :  하드마진이 가진 한계를 개선하고자 나온 개념으로, 완벽하게 분류하는 초평면을 찾는 것이 아니라 어느 정도의 오분류를 허용하는 방식이다. 소프트마진에서는 오분류를 허용하고 이를 고려하기 위해 slack variable을 사용하여 해당 결정경계로부터 잘못 분류된 데이터의 거리를 측정한다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|\left\|w\right\|\right\|^2 + C\sum_{i=1}^{n}\xi_i$

![](./images/smargin.png)

- 비선형분류 : 선형분리가 불가능한 입력공간을 선형분리가 가능한 고차원 특성공간으로 보내 선형분리를 진행하고 그 후 다시 기존의 입력공간으로 변환하면 비선형 분리를 하게 된다.<br><br>
![](./images/nlsvm.png)

<br>

	from sklearn.datasets import make_moons
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import PolynomialFeatures

	polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),("scaler", StandardScaler()),("svm_clf", LinearSVC(C=10, loss="hinge", max_iter=2000, random_state=42))])
	polynomial_svm_clf.fit(X, y)

<br>

입력공간을 특성공간으로 변환하기 위해서 mapping function을 사용한다<br>
$\Phi(x) = Ax$<br><br>
고차원의 특성공간으로 변환하고 목적함수에 대한 문제를 푸는 것이 간단한 차원에서는 가능하나 그 차수가 커질수록 계산량의 증가하는 것을 다시 해결하고자 나오는 개념이 커널트릭(Kernel trick) : 비선형 분류를 하기 위해 차원을 높여줄 때마다 필요한 엄청난 계산량을 줄이기 위해 사용하는 커널 트릭은 실제로는 데이터의 특성을 확장하지 않으면서 특성을 확장한 것과 동일한 효과를 가져오는 기법<br>
$k(x_i, x_j) =\Phi(x_i)^T\Phi(x_j)$<br><br>
확장된 특성공간의 두 벡터의 내적만을 계산하여 고차원의 복잡한 계산 없이 커널 함수를 사용하여 연산량을 간단하게 해결할 수 있다. 가장 성능이 좋고 많이 사용되는 것이 가우시안 RBF(Radial basis function)으로 두 데이터 포인트 사이의 거리를 비선형 방식으로 변환하여 고차원 특징 공간에서 분류 문제를 해결하는 데 사용.<br><br>
$k(x,y) = e^{-\frac{-\left\|x_i-x_j\right\|^2}{2\sigma^2}}$<br><br>
직접 차수를 정하는 방식(Polynomial) : $k(x,y) = (1+x^Ty)^p$<br>
신경망 학습(Signomail) : $k(x,y) = tanh(kx_ix_j-\delta)$<br>

![](./images/hnlsvm.png)

<br>  

## 서포트 벡터 회귀(Support Vector Regression, SVR)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html<br>
▣ 모델식 : https://scikit-learn.org/stable/modules/svm.html#svr<br>
▣ 정의 : 데이터 포인트들을 초평면 근처에 배치하면서, 허용 오차 $ϵ$ 내에서 예측 오차를 최소화하는 것.<br>



	from sklearn.svm import SVR
 
 	svr = SVR(kernel='rbf', gamma='auto')
	svr.fit(xtrain, ytrain)

	score = svr.score(xtest, ytest)
	print("R-squared: ", score)

<br> 

## 서포트 벡터 분류(Support Vector Classification, SVC)
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC<br>
▣ 모델식 : https://scikit-learn.org/stable/modules/svm.html#svc<br>
▣ 정의 : 두 클래스(또는 다수의 클래스)를 분류하기 위해 최대 마진을 가지는 초평면을 찾는 것.<br>

	import sklearn.svm as svm

 	# 선형일 경우
	svm_clf =svm.SVC(kernel = 'linear')
 	# 비선형일 경우
 	svm_clf =svm.SVC(kernel = 'rbf')

	# 교차검증
	scores = cross_val_score(svm_clf, X, y, cv = 5)
 	scores.mean()

<br>

# [3] 결정 트리(Decision Tree)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/index.html<br>
▣ 정의 : 어떤 항목에 대한 관측값과 목표값을 연결시켜주는 예측 모델로, 대표적인 지도학습 분류 모델이며, 스무고개와 같이 질문에 대하여 '예' 또는 '아니오'를 결정하여 트리 구조를 나타낸다.<br> 

![](./images/tree.png)

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 시각화를 통한 해석의 용이성(나무 구조로 표현되어 이해가 쉬움, 새로운 개체 분류를 위해 루트 노드부터 끝 노트까지 따라가면 되므로 분석 용이) | 휴리스틱에 근거한 실용적 알고리즘으로 학습용 자료에 의존하기에 전역 최적화를 얻지 못할 수도 있음(검증용 데이터를 활용한 교차 타당성 평가를 진행하는 과정이 필요) |
| 데이터 전처리, 가공작업이 불필요 | 자료에 따라 불안정함(적은 수의 자료나 클래스 수에 비교하여 학습 데이터가 적으면 높은 분류에러 발생) | 
| 수치형, 범주형 데이터 모두 적용 가능 | 각 변수의 고유한 영향력을 해석하기 어려움 | 
| 비모수적인 방법으로 선형성, 정규성 등의 가정이 필요없고 이상값에 민감하지 않음 | 자료가 복잡하면 실행시간이 급격하게 증가함 | 
| 대량의 데이터 처리에도 적합하고 모형 분류 정확도가 높음 | 연속형 변수를 비연속적 값으로 취급하여 분리 경계점에서는 예측오류가 매우 커지는 현상 발생 | 

![](./images/trees.png)

<br>

# 결정 트리 회귀(Decision Tree Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html<br>
▣ 정의 : 데이터에 내재되어 있는 패턴을 비슷한 수치의 관측치 변수의 조합으로 예측 모델을 나무 형태로 만든다.<br>
▣ 모델식 : $\widehat{f}(x) = \sum_{m=1}^{M}C_mI((x_1,x_2)\in R_m)$<br>
비용함수(cost function)를 최소로 할때 최상의 분할 : 데이터를 M개로 분할($R_1,R_2,...R_M$)<br> 
$\underset{C_m}{min}\sum_{i=1}^{N}(y_i-f(x_i))^2=\underset{C_m}{min}\sum_{i=1}^{N}(y_i-\sum_{m=1}^{M}C_mI(x\in R_m))^2$<br>
각 분할에 속해 있는 y값들의 평균으로 예측했을때 오류가 최소화 : $\widehat{C}_m=ave(y_i|x_i\in R_m)$<br>

	from sklearn.tree import DecisionTreeRegressor
 	from sklearn.metrics import mean_squared_error
 
 	# 결정 트리 회귀 모델 생성 및 학습 (최대 깊이 5)
	tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
	tree_reg.fit(X_train, y_train)

	# 테스트 데이터에 대한 예측
	y_pred = tree_reg.predict(X_test)

	# 모델 성능 평가 (평균 제곱 오차)
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse}")

<br>

# 결정 트리 분류(Decision Tree Classification)
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html<br>
▣ 정의 : 데이터에 내재되어 있는 패턴을 비슷한 범주의 관측치 변수의 조합으로 분류 모델을 나무 형태로 만든다.<br>
▣ 모델식 : $\widehat{f(x)} = \sum_{m=1}^{M}k(m)I((x_1,x_2)\in R_m)$<br>
끝노드(m)에서 클래스(k)에 속할 관측치의 비율 : $\widehat{P_{mk}}=\frac{1}{N_m}\sum_{x_i\in R_m}^{}I(y_i=k)$<br>
끝노드 m으로 분류된 관측치 : $k(m) = \underset{k}{argmax}\widehat{P_{mk}}$<br><br>
▣ 비용함수(불순도 측정) : 불순도(Impurity)가 높을수록 다양한 클래스들이 섞여 있고, 불순도가 낮을수록 특정 클래스에 속한 데이터가 명확<br>
(1) 오분류율(Misclassification rate, Error rate) : 분류 모델이 잘못 분류한 샘플의 비율로, 전체 샘플 중에서 실제 값과 예측 값이 일치하지 않는 샘플의 비율을 나타낸다.(0에서 100 사이의 값, 0%: 모델이 모든 샘플을 완벽하게 예측, 100%: 모델이 모든 샘플을 잘못 예측)<br><br>
$\frac{FP+FN}{TP+TN+FP+FN}$ 
###### FP(False Positive) : 실제 값이 Negative인데 Positive로 예측, FN(False Negative) : 실제 값이 Positive인데 Negative로 예측, TP(True Positive) : 실제 값이 Positive이고 Positive로 올바르게 예측, TN(True Negative) : 실제 값이 Negative이고 Negative로 올바르게 예측<br>
(2) 지니계수(Gini Coefficient) : 데이터셋이 얼마나 혼합되어 있는지를 나타내는 불순도의 측정치(0에서 0.5 사이의 값, 0: 데이터가 완벽하게 한 클래스에 속해 있음을 의미하며, 불순도가 전혀 없는 상태, 0.5: 두 개의 클래스가 완벽하게 섞여 있는 상태)<br>
$Gini(p)=1-\sum_{i=1}^{n}p_i^2$<br><br>
(3) 엔트로피(Entropy) : 확률 이론에서 온 개념으로 불확실성 또는 정보의 무질서를 측정하는 또 다른 방식으로, 데이터가 얼마나 혼란스럽고 예측하기 어려운지를 측정(0에서 1 사이의 값, 0 : 데이터가 완벽하게 한 클래스에 속해 있으며, 불확실성이 없는 상태, 1 : 데이터가 완전히 섞여 있고, 가장 큰 불확실성을 가지고 있다.<br>
$Entropy(p) = -\sum_{i=1}^{n}p_ilog_2p_i$<br> 

▣ 유형 :  ID3, CART
 - ID3 : 모든 독립변수가 범주형 데이터인 경우에만 분류가 가능하다. 정보획득량(Infomation Gain)이 높은 특징부터 분기해나가는데 정보획득량은 분기전 엔트로피와 분기후 엔트로피의 차이를 말한다.(엔트로피 사용)<br><br>
$IG(S, A) = E(S) - E(S|A)$<br>
 - CART : Classification and Regression Tree의 약자로, 이름 그대로 분류와 회귀가 모두 가능한 결정트리 알고리즘으로 yes 또는 no 두 가지로 분기한다.(지니계수 사용)<br><br> 
$f(k,t_k) = \frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}$<br>

<br>

	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 	#결정 트리 분류 모델 생성 (최대 깊이 3으로 설정)
	clf = DecisionTreeClassifier(max_depth=3, random_state=42)
	clf.fit(X_train, y_train)  # 학습 데이터를 이용해 모델 학습

	#테스트 데이터에 대한 예측
	y_pred = clf.predict(X_test)

	#정확도 출력
	accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
	print(f"Accuracy: {accuracy * 100:.2f}%")  # 정확도 출력


<br>

	(개별 트리 모델의 단점)	
 	계층적 구조로 인해 중간에 에러가 발생하면 다음 단계로 에러가 계속 전파
  	학습 데이터의 미세한 변동에도 최종결과에 큰 영향
   	적은 개수의 노이즈에도 큰 영향
	나무의 최종 노드 개수를 늘리면 과적합 위함(Low Bias, Large Variance)

	(해결방안) 랜덤 포레스트(Random forest)

<br> 

# [4] 랜덤 포레스트(Random Forest)  
▣ 가이드 : https://scikit-learn.org/stable/modules/ensemble.html#random-forests<br>
▣ 정의 : 분류와 회귀에 사용되는 지도학습 알고리즘으로 여러 개의 의사결정나무(Decision Tree)를 조합한 **앙상블 학습(ensemble learning)** 을 적용한 모델이다. 여러개의 Training data를 생성하여 각 데이터마다 개별 의사결정나무모델을 구축하는 배깅(bootstrap aggregation, bagging)과 의사결정 모델 구축시 변수를 무작위로 선택하는 Random subspace가 특징.<br>
▣ 모델식 : $\widehat{y}=\frac{1}{N}\sum_{i=1}^{N}T_i(X)$ ($N$ : 결정트리의 수, $T_i(X)$ : 각 결정트리 $i$가 입력값 $X$에 대해 예측한 값)

![](./images/Bootstrap.png)
출처: https://www.researchgate.net/figure/Schematic-of-the-RF-algorithm-based-on-the-Bagging-Bootstrap-Aggregating-method_fig1_309031320<br>


| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 모델이 단순, 과적합이 잘 일어나지 않음 | 여러개의 결정트리 사용으로 메모리 사용량 큼 |
| 새로운 데이터에 일반화가 용이함 | 고차원 및 희소 데이터에 잘 작동하지 않음 |

# 랜덤 포레스트 회귀(Random Forest Regression)  
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#randomforestregressor<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor<br>
▣ 정의 : 랜덤 포레스트 회귀 모델은 각 트리가 예측한 값들의 평균을 통해 최종 예측값을 도출하는 모델로, 다수결 대신, 트리에서 얻은 예측값의 평균을 사용하여 연속적인 값을 예측한다.<br>
▣ 모델식 : $\widehat{y}= \frac{1}{B}\sum_{i=1}^{B}T_i(x)$<br>
###### $T_i(x)$: 입력 데이터 𝑥에 대한 𝑖번째 결정 트리의 예측값, B: 전체 트리의 개수


	from sklearn.ensemble import RandomForestRegressor
 
 	# 트리 개수를 변화시키며 모델 학습 및 평가
	for iTrees in nTreeList:
    		depth = None  # 트리 깊이 제한 없음
    		maxFeat = 4  # 사용할 최대 특징 수
    		# 랜덤 포레스트 회귀 모델 생성 및 학습
    		wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
			max_depth=depth, max_features=maxFeat,
			oob_score=False, random_state=531)
    		wineRFModel.fit(xTrain, yTrain)  # 모델 학습
    		# 테스트 데이터에 대한 예측값 계산
    		prediction = wineRFModel.predict(xTest)
    		# MSE 계산 및 누적
    		mseOos.append(mean_squared_error(yTest, prediction))
     
	# MSE 출력
	print("MSE")
	print(mseOos)


<br>

# 랜덤 포레스트 분류(Random Forest Classification)    	  	
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#randomforestclassifier<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier<br>
▣ 정의 : 랜덤 포레스트 분류 모델은 다수의 의사결정나무(Decision Trees)를 기반으로 한 앙상블 모델로, 각 나무는 독립적으로 클래스를 예측한 후 다수결 투표를 통해 최종 클래스를 결정한다.<br>
▣ 모델식 : $\widehat{y}=mode(T_1(x),T_2(x),...,T_B(x))$<br>
###### $T_i(x)$: 입력 데이터 𝑥에 대한 𝑖번째 결정 트리의 예측값, B: 전체 트리의 개수, mode 함수 : 다수결 투표방식

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn import datasets

	# 붓꽃 데이터셋 로드
	iris = datasets.load_iris()

	# 독립 변수와 종속 변수 분리
	X = iris.data
	y = iris.target

	# 학습용 데이터와 테스트용 데이터 분리
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 랜덤 포레스트 모델 초기화
	model = RandomForestClassifier()

	# 모델 학습
	model.fit(X_train, y_train)

	# 테스트 데이터 예측
	y_pred = model.predict(X_test)

	# 정확도 계산
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)
 
<br>
