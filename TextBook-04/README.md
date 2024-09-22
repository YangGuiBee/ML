#  04 : 지도 학습 (Supervised Learning, SL) : 회귀 (regression)

---

	[1] 선형 회귀 (Linear Regression)
	[2] 다항 회귀 (Polynomial Regression)
 	[2-1] 분위수 회귀 (Quantile Regression)
  	[2-2] 단계적 회귀 (Stepwise Regression)
   	[2-3] 포아송 회귀 (Poisson Regression)
 	[3] 정규화된 회귀 (Regularized Regression), 벌점 회귀 (Penalized Regression)
	[3-1] 릿지 회귀 (Ridge Regression)
	[3-2] 라쏘 회귀 (Lasso Regression)
	[3-3] 엘라스틱넷 회귀 (Elastic Net Regression)

---

	로지스틱 회귀 (Logistic Regression) → 분류	
	k-최근접 이웃 회귀(k-Nearest Neighbors Regression) → 분류+회귀 
	서포트 벡터 회귀 (Support Vector Regression, SVR) → 분류+회귀 
	결정 트리 회귀 (Decision Tree Regression) → 분류+회귀 
	랜덤 포레스트 회귀 (Random Forest Regression) → 분류+회귀   
 	주성분 회귀 (Principal Component Regression) → 차원축소   

---

**지도 학습**은 주어진 입력값($X$)에 대하여 신뢰성 있는 출력값($y$)을 출력하는 함수를<br> 
현재 가지고 있는 데이터(학습 데이터 $X$, $y$)로부터 학습하는 과정이다.<br>
수식을 이용하여 표현하면, 현재 가지고 있는 학습데이터 $(X, y)$로부터 $y = f(X)$를 만족하는<br> 
여러 함수 $f$중에서 가장 최적의(주어진 Task에 따라 달라짐) $f$를 찾는 과정이라고 할 수 있다.<br>
출력 변수 $y$가 최적 함수 $f$를 찾도록 지도해주는 역할을 한다고 해서 지도 학습이라고 한다.<br>

지도 학습은 **회귀(Regression)** 와 **분류(Classification)** 로 구분된다.<br>
회귀 모델은 예측값으로 연속적인 값을 출력하고, 분류 모델은 예측값으로 이산적인 값을 출력한다.<br> 

예를 들어, 도미와 빙어의 길이와 무게 데이터를 통해 도미 여부를 식별하는 것은 분류(출력변수 : 범주형),<br> 
도미의 길이 데이터를 통해 도미의 무게를 예측하는 것은 회귀(출력변수 : 연속형)이다.<br>

<br>

# [1] 선형 회귀 (Linear Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/linear_model/index.html<br>
종속변수 y(예상값)과 독립변수(설명변수) X와의 선형 상관 관계를 모델링하는 회귀<br>
<br>
$y = wx + b$ <br>
$y = w_1x + w_0$ <br>

![](./images/LinearRegression.gif)
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f
<br><br>
모델 추정을 위해서, 예측값과 실제관측값인 잔차의 제곱을 최소화하는 최소제곱법(OLS)을 사용<br>
<img width ='500' height = '400' src = 'https://github.com/YangGuiBee/ML/blob/main/TextBook-04/images/LRd.png'>
<br>
출처 : https://blog.csdn.net/Amy_mm/article/details/79989722
<br>

선형회귀는 학습을 통해 잔차 제곱들의 합인 RSS(Residual Sum of Squares)를 최소로 하는 회귀계수($W_0$과 $W_1$)를 찾는 것이 핵심.<br>
잔차제곱합(RSS)을 최소화하는 방법을 최소제곱회귀 혹은 Ordinary Least Squares(OLS) 라고 부른다.<br>
![](./images/RSSd.svg)
<br>

	from sklearn.linear_model import LinearRegression

	lr = LinearRegression()
	lr.fit(train_input, train_target)

	print(lr.predict(([50]))

	# coef_ : 기울기, intercept_ : 절편
	print(lr.coef_, lr.intercept_)

'선형'은 독립변수가 1차항으로써 2차원에서는 직선형태로, 3차원 공간에서는 평면으로 나타나며,<br> 
독립변수에 지수항이 있으면 '비선형'이 된다.<br> 
독립변수 X가 2개 이상인 회귀는 다중선형회귀 (Multiple  Linear Regression)<br>
$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$ <br>

<br>

# [2] 다항 회귀 (Polynomial Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions<br>
독립변수와 종속변수가 선형관계가 아닌 비선형 회귀(Non-linear Regression)<br>
직선이 아닌 곡선 형태의 관계의 경우, 독립변수에 제곱이나 로그(log) 등을 취해 보면서 실시하는 모델링<br>
$y = w_1x + w_2x^2 + ... + w_nx^n + b$ <br>
<br>
![](./images/PL.png)


<br>

	입력데이터의 Feature들이 너무 많은 경우(Feature수에 비해 관측치 수가 적은 경우) 과적합이 발생
	→ 
	(해결방안1) 데이터를 더 수집하거나 불필요한 Features들을 제거
	(해결방안2) 가중치(회귀계수)에 페널티 값을 적용하는 규제(Regularization)를 통해 
 	            Feature들에 곱해지는 가중치가 커지지 않도록 제한

<br>

# [3-1] 릿지 회귀 (Ridge Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification<br>
최소제곱추정치(OLS)가 편향되지 않더라도 분산이 커서 관측값이 실제값에서 크게 벗어나는 다중공선성(multicollinearity)이 발생할 경우, 회귀 분석 추정치에 치우침 정도를 추가하여 표준오차를 줄이기 위해 사용<br>
모델의 설명력에 기여하지 못하는 독립변수의 회귀계수 크기를 0에 근접하도록 축소시키는 회귀<br>
L2-norm 페널티항을 통해 일반 선형회귀 모델에 페널티를 부과하는 방법으로 회귀계수를 축소<br>
(L2 norm : 실제값과 예측값의 오차의 제곱의 합)

<br>

# [3-2] 라쏘 회귀 (Lasso Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#lasso<br>
릿지회귀모델과 다르게 설명력에 기여하지 못하는 독립변수의 회귀계수를 0으로 만드는 회귀<br>
L1-norm 패널티항으로 회귀모델에 패널티를 부과함으로써 회귀계수를 축소<br>
(L1 norm : 실제값과 예측값의 오차의 절대값의 합)

<br>

# [3-3] 엘라스틱넷 회귀 (Elastic Net Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net<br>
릿지와 라쏘의 결합<br>
L1-norm 과 L2-norm을 모두 이용하여 패널티를 부과하여 회귀모델을 생성<br>

<br>

![](./images/L1L2.png)
<br>
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f


