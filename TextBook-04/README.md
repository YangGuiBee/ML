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
'선형'은 독립변수가 1차항으로써 2차원에서는 직선형태로, 3차원 공간에서는 평면으로 나타난다.<br> 
<br>
$y = wx + b$ <br>
​​
<br>

![](./images/LinearRegression.gif)
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f
<br><br>
모델 추정을 위해서, 예측값과 실제관측값인 잔차의 제곱을 최소화하는 최소제곱법(OLS)을 사용<br>
<img width ='500' height = '400' src = 'https://github.com/YangGuiBee/ML/blob/main/TextBook-04/images/LRd.png'>
<br>
출처 : https://blog.csdn.net/Amy_mm/article/details/79989722
<br>

선형회귀는 학습을 통해 잔차 제곱들의 합인 RSS(Residual Sum of Squares)를 최소로 하는<br> 
회귀계수($W_0$과 $W_1$)를 찾는 것이 핵심.<br>
잔차제곱합(RSS)을 최소화하는 방법을 최소제곱회귀 혹은 Ordinary Least Squares(OLS) 라고 부른다.<br>
![](./images/RSSd.svg)
<br>

	from sklearn.linear_model import LinearRegression

	lr = LinearRegression()
	lr.fit(train_input, train_target)

	print(lr.predict(([50]))

	# coef_ : 기울기(w1), intercept_ : 절편(w0)
	print(lr.coef_, lr.intercept_)


독립변수가 1개이고, 모델이 독립변수와 회귀계수에 대하여 선형인 경우를 **단순선형회귀 (simple linear regression)** 라 부른다.<br>
$y = w_1x + w_0$ <br>
$y_i = β_1x_i + β_0 + ϵ_i$<br>
$y_i$ : i번째 반응변수 값<br>
$x_i$ : i번째 설명변수 값<br>
$β_0$ : 절편 회귀계수<br>
$β_1$ : 기울기 회귀계수<br>
$ϵ_i$ : i번째 측정된 $y_i$의 오차 성분<br>
$E[ϵ_i]=0, Var(ϵ_i)=σ^2, E[ϵ_iϵ_j] = δ_{ij}$<br>
<br>
독립변수 X가 2개 이상인 회귀는 **다중회귀 (Multiple Regression)** 라고 한다.<br>
$y = w_1x_1 + w_2x_2 + ... + w_nx_n + w_0$ <br>
$y_i = β_0 + β_1x_{i1} + β_2x_{i2} + ... + β_kx_{ik} + ϵ_i$<br>
$y_i$ : i번째 관측치, $ϵ_i$ : 이때의 오차항, $x_{ij}$ : 독립변수로 known value<br>
$β_j$ : 추정하고자하는 값인 회귀계수로 $0≤j≤k$ 사이의 값<br>
N개의 샘플에 대하여 확장한 후, vector-matrix 형태로 표기하면,<br>
<img width ='500' src = 'https://github.com/YangGuiBee/ML/blob/main/TextBook-04/images/vectorMX.png'><br>
$e∼N(0,σ^2I_N)$<br>

	import pandas as pd
	import matplotlib.pyplot as plt 
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression
	
	df = pd.read_csv('https://raw.githubusercontent.com/YangGuiBee/ML/main/TextBook-04/manhattan.csv')
	df = df.drop(['neighborhood','borough','rental_id'], axis=1)
	print(df.columns)
	print(len(df.columns))
	
	X = df [['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor',
       	'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer',
	'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio','has_gym']]
	y = df [['rent']]       
 
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
	mlr = LinearRegression()
	mlr.fit(X_train, y_train)
	
	my_apartment = [[1,2,620,16,1,98,1,0,1,0,0,1,1,0]]
	my_predict = mlr.predict(my_apartment)
	my_predict
	
	y_predict = mlr.predict(X_test)
	
	plt.scatter(y_test,y_predict,alpha=0.4)
	plt.xlabel('Actual Rent')
	plt.ylabel('Predicted Rent')
	plt.title('Multiple Linear Regression')
	plt.show()
 
	# 특성별 상관분석
	#plt.scatter(df[['size_sqft']],df[['rent']], alpha = 0.4)
	#plt.show()
	
	#r2(coefficient of determination) : 결정계수 = 1 - (RSS/TSS)
	#RSS(Residual Sum of Square) : 잔차의 제곱의 평균으로 직선이 미처 y에 대해 설명하지 못한 변화량
	#TSS(Total Sum of Squares) : y값의 총 변화량
	
	mlr.score(X_train, y_train)

<br>

# [2] 다항 회귀 (Polynomial Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions<br>
독립변수와 종속변수가 선형관계가 아닌 비선형 회귀(Non-linear Regression)<br>
직선이 아닌 곡선 형태의 관계의 경우, 독립변수에 제곱이나 로그(log) 등을 취해 보면서 실시하는 모델링<br>
$y = w_1x_1 + w_2x_2^2 + ... + w_nx_n^n + w_0$ <br>
<br>
![](./images/PolynomialFeatures.png)


	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt 
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score
	
	df = pd.read_csv('https://raw.githubusercontent.com/YangGuiBee/ML/main/TextBook-04/housing.data.txt',
                 header=None, sep='\s+')

	df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	df.head()
	
	X = df[['LSTAT']].values
	y = df['MEDV'].values
	
	regr = LinearRegression()

	# 이차, 삼차 다항식 특성을 만듭니다
	quadratic = PolynomialFeatures(degree=2)
	cubic = PolynomialFeatures(degree=3)
	X_quad = quadratic.fit_transform(X)
	X_cubic = cubic.fit_transform(X)

	# 학습된 모델을 그리기 위해 특성 범위를 만듭니다
	X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
	
	regr = regr.fit(X, y)
	y_lin_fit = regr.predict(X_fit)
	linear_r2 = r2_score(y, regr.predict(X))
	
	regr = regr.fit(X_quad, y)
	y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
	quadratic_r2 = r2_score(y, regr.predict(X_quad))
	
	regr = regr.fit(X_cubic, y)
	y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
	cubic_r2 = r2_score(y, regr.predict(X_cubic))
		
	# 결과 그래프를 그립니다
	plt.scatter(X,y,label='Training points', color='lightgray')
	plt.plot(X_fit,y_lin_fit,label='Linear(d=1),$R^2=%.2f$' % linear_r2,color='blue',lw=2,linestyle=':')
	plt.plot(X_fit,y_quad_fit,label='Quadratic(d=2),$R^2=%.2f$' % quadratic_r2,color='red',lw=2,linestyle='-')
	plt.plot(X_fit,y_cubic_fit,label='Cubic(d=3),$R^2=%.2f$' % cubic_r2,color='green',lw=2,linestyle='--')
	plt.xlabel('% lower status of the population [LSTAT]')
	plt.ylabel('Price in $1000s [MEDV]')
	plt.legend(loc='upper right')
	plt.show()

<br>

# [2-1] 분위수 회귀 (Quantile Regression)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html<br>
반응 변수의 조건부 분위수를 모델링 하는 모델<br>
1) 선형 회귀 조건이 충족되지 않는 경우<br>
2) 오차의 분산이 큰 경우<br>
3) Robust한 결과를 위하여<br>
4) 많은 이상치의 영향을 줄이기 위하여<br>
5) 점 추정이 아닌 구간추정을 통해 결과의 정확도를 높이기 위하여<br>
6) 반응변수의 스프레드를 같이 살펴보기 위하여<br>
7) 회귀곡선에 대한 설득력을 높이기 위하여<br>
사용되는 선형 회귀의 확장 버전<br>
<br>
보통 OLS 회귀는 조건부 평균값을 모델링하는 반면 분위수 회귀는 조건부 분위수를 모델링하고<br>
조건부 분위수를 모델링하기 위해 Pinball loss를 사용<br>
기존의 조건부 평균 값 예측이 아닌 조건부 분위수 값을 예측하는 문제로 풀이 될 수 있다.<br>

$Q_{\tau}(y_{i}) = \beta_{0}(\tau) + \beta_{1}(\tau)x_{i1} + \cdots + \beta_{p}(\tau)x_{ip}$<br>

최적의 분위수 방정식을 찾기 위한 과정은 중위수절대편차인 MAD(Median Absolute Deviation) 값을 최소화함으로써 찾을 수 있다.<br>
$MAD = \frac{1}{n} \sum_{i=1}^{n} \rho_{\tau}(y_{i} - (\beta_{0}(\tau) + \beta_{1}(\tau)x_{i1} +\cdots +\beta_{p}(\tau)x_{ip}))$<br>
 
ρ함수는 오차의 분위수와 전체적인 부호에 따라 오차에 비대칭 가중치를 부여하는 체크 함수<br>
$\rho_{\tau}(u) = \tau\max(u,0) + (1-\tau)\max(-u,0)$<br>
<br>

	from sklearn.linear_model import LinearRegression
	lm_model = LinearRegression()
	lm_model.fit(X,y)
	y_pred = lm_model.predict(X)
	# 분위수 회귀 모형 구축
	mod = smf.quantreg('Price ~ Area', house_data)
	# 각 분위수에 따른 분위수 회귀 값 저장
	quantiles = np.arange(.05,.96,.1) # quantiles = [.05,.15,.25,...,.95]
	
	def fit_model(q):
	  res = mod.fit(q=q)
	  return [q, res.params['Intercept'], res.params['Area']] + \
	  res.conf_int().loc['Area'].tolist()
	  
	models = [fit_model(x) for x in quantiles]
	models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

	# 비교를 위해 최소 기존의 선형 회귀 값도 저장
	ols = smf.ols('Price ~ Area', house_data).fit()
	ols_ci = ols.conf_int().loc['Area'].tolist()
	ols = dict(a = ols.params['Intercept'], b = ols.params['Area'], lb = ols_ci[0], ub = ols_ci[1])

	print(models)
	print(ols)

<br>

# [2-2] 단계적 회귀 (Stepwise Regression)
여러 독립변수 중에서 종속변수를 가장 잘 설명하는 변수들을 선택하는 방법<br>
독립 변수들을 자동으로 모델에 추가하거나 제거하여 최적의 모델을 탐색(변수의 추가나 제거가 통계적으로 유의미한지 여부에 따라 이루어짐)<br>
예를 들어, 변수를 추가할 때마다 F 통계량이유의미하게 증가하는지 확인하거나, 제거할 때마다 변수의 t 통계량이 유의미하게 감소하는지 확인함.<br> 
장점: 자동으로 변수를 선택하므로 모델이 데이터에 더 잘 맞을 가능성이 있음<br>
**위계적 회귀 (Hierarchical Regression)** 는 독립 변수들을 미리 정의한 순서에 따라 모델에 추가하는 것으로,<br>
이론적으로 중요한 변수부터 시작하여 덜 중요한 변수를 차례로 추가하는 방식<br>
장점: 이론적 근거에 따라 변수를 추가하므로 결과 해석이 이론적으로 타당함.<br>

<br>

# [2-3] 포아송 회귀 (Poisson Regression)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html<br>
종속변수가 포아송 분포를 따르는 경우에 사용되며, 이산형 카운트 데이터를 모델링하는 데 적합<br>
포아송 분포(Poisson Distribution)는 단위(한정된) 시간이나 공간에서 발생하는 평균적인 사건의 횟수(λ)를 바탕으로 특정 횟수의 사건이 발생할 확률을 설명하는 분포<br>

**포아송 과정(Poisson process)** <br>
1) 정상성(stationarity): 현상이 발생하는 횟수의 분포는 시작 시각과 관계없음. 즉, 
$N_t$의 분포와 $N_{s+t}−N_S$의 분포가 같고 $N_0=0$이다.
2) 독립 증분성(independent increment): 시각 0부터 $t$까지 현상이 발생하는 횟수와 시각 $t$후부터 $t+h(h>0)$까지의 발생 횟수는 서로 독립(즉, $N_t$와 $N_{t+h}−N_t$는 서로 독립)
3) 비례성(propertionality): 짧은 시간 동안에 현상이 한 번 발생할 확률은 시간에 비례.
$P(N_h=1)=λh+o(h),h→0$
※ λ는 양의 비례상수, o(h)는 $lim_{h→0}o(h)/h=0$
4) 희귀성(rareness): 짧은 시간 동안에 현상이 두 번 이상 발생할 확률은 매우 작음.
$P(N_h≥2)=o(h),h→0$<br>

**포아송 확률변수 $X$의 확률밀도함수(probability mass function)** <br>
$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}$<br>

**포아송 회귀 적용 사례** <br>
1) 일정 주어진 시간 동안에 방문하는 고객의 수<br>
2) 일정 주어진 생산시간 동안 발생하는 불량 수<br>
3) 하룻동안 발생하는 출생자 수<br>
4) 어떤 시간 동안 톨게이트를 통과하는 차량의 수<br>
5) 어떤 페이지에 있는 오타의 발생률<br>
6) 어떤 특정 면적의 삼림에서 자라는 소나무의 수<br>

<br>

	import numpy as np
	from scipy.stats import poisson
	import numpy as np
	import seaborn as sns
	from scipy.special import factorial
	
	np.random.seed(123)
	poisson.rvs(mu = 1, size = 10)
	pal_brbg = sns.color_palette("BrBG", 6)
	
	x = np.arange(0, 11)
	for n_lambda in range(1, 6):
	    y = np.exp(-n_lambda) * np.power(n_lambda, x) / factorial(x)
	    plt.plot(x, y, color = pal_brbg[n_lambda - 1], label=f"λ = {n_lambda}")
	    plt.scatter(x, y, color = pal_brbg[n_lambda - 1])
    	
	plt.ylabel("Probability")
	plt.title(f"Poisson Distribution (λ = [1, 5])")
	plt.xticks(x)
	plt.grid(axis = "y", linestyle = "--", color = "#CCCCCC")
	plt.legend(loc="upper right")
	plt.show()


<br>

# 기타 회귀 분석
<br>

**매개회귀(Mediation Regression)**
독립변수와 종속변수 간의 관계가 다른 변수(매개변수)에 의해 어떻게 매개되는지 분석
<br>

**조절회귀(Mederation Regression)**
독립변수와 종속변수 간의 관계가 다른 변수(조절변수)에 의해 어떻게 매개되는지 분석
<br>

**시계열 회귀 (Time Series Regression)**
시간에 따라 변하는 데이터에 적용되며, 시간 요인을 고려하여 독립 변수와 종속변수간의 관계를 모델링


<br>

	입력데이터의 Feature들이 너무 많은 경우(Feature수에 비해 관측치 수가 적은 경우) 과적합이 발생
	→ 
	(해결방안1) 데이터를 더 수집하거나 불필요한 Features들을 제거
	(해결방안2) 가중치(회귀계수)에 페널티 값을 적용하는 규제(Regularization)를 통해 
 	            Feature들에 곱해지는 가중치가 커지지 않도록 제한

<br>

# [3-1] 릿지 회귀 (Ridge Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html<br>
최소제곱추정치(OLS)가 편향되지 않더라도 분산이 커서 관측값이 실제값에서 크게 벗어나는 다중공선성(multicollinearity)이 발생할 경우, 회귀 분석 추정치에 치우침 정도를 추가하여 표준오차를 줄이기 위해 사용<br>
모델의 설명력에 기여하지 못하는 독립변수의 회귀계수 크기를 0에 근접하도록 축소시키는 회귀<br>
L2-norm 페널티항을 통해 일반 선형회귀 모델에 페널티를 부과하는 방법으로 회귀계수를 축소<br>
(L2 norm : 실제값과 예측값의 오차의 제곱의 합)<br>

	from sklearn.linear_model import Ridge
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score

	ridge = Ridge(alpha=1.0)
	ridge.fit(X_train, y_train)
	y_train_pred = ridge.predict(X_train)
	y_test_pred = ridge.predict(X_test)
	print(ridge.coef_)
	
	print('훈련 MSE: %.3f, 테스트 MSE: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('훈련 R^2: %.3f, 테스트 R^2: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

<br>

# [3-2] 라쏘 회귀 (Lasso Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#lasso<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html<br>
릿지회귀모델과 다르게 설명력에 기여하지 못하는 독립변수의 회귀계수를 0으로 만드는 회귀<br>
L1-norm 패널티항으로 회귀모델에 패널티를 부과함으로써 회귀계수를 축소<br>
(L1 norm : 실제값과 예측값의 오차의 절대값의 합)<br>

	from sklearn.linear_model import Lasso
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	
	lasso = Lasso(alpha=0.1)
	lasso.fit(X_train, y_train)
	y_train_pred = lasso.predict(X_train)
	y_test_pred = lasso.predict(X_test)
	print(lasso.coef_)
	
	print('훈련 MSE: %.3f, 테스트 MSE: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
	print('훈련 R^2: %.3f, 테스트 R^2: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

<br>

# [3-3] 엘라스틱넷 회귀 (Elastic Net Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html<br>
릿지와 라쏘의 결합<br>
L1-norm 과 L2-norm을 모두 이용하여 패널티를 부과하여 회귀모델을 생성<br>

	from sklearn.linear_model import ElasticNet
	elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

<br>

![](./images/L1L2.png)
<br>
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f


