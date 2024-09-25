#  04 : 지도 학습 (Supervised Learning, SL) : 회귀 (regression)
**지도 학습**은 주어진 입력값($X$)에 대하여 신뢰성 있는 출력값($y$)을 출력하는 함수를<br> 
현재 가지고 있는 데이터(학습 데이터 $X$, $y$)로부터 학습하는 과정이다.<br>
수식을 이용하여 표현하면, 현재 가지고 있는 학습데이터 $(X, y)$로부터 $y = f(X)$를 만족하는<br> 
여러 함수 $f$중에서 가장 최적의(주어진 Task에 따라 달라짐) $f$를 찾는 과정이라고 할 수 있다.<br>
출력 변수 $y$가 최적 함수 $f$를 찾도록 지도해주는 역할을 한다고 해서 지도 학습이라고 한다.<br>

지도 학습은 **회귀(Regression)** 와 **분류(Classification)** 로 구분된다.<br>
회귀 모델은 예측값으로 연속적인 값을 출력하고, 분류 모델은 예측값으로 이산적인 값을 출력한다.<br> 

예를 들어, 도미와 빙어의 길이와 무게 데이터를 통해 도미 여부를 식별하는 것은 분류(출력변수 : 범주형),<br> 
도미의 길이 데이터를 통해 도미의 무게를 예측하는 것은 회귀(출력변수 : 연속형)이다.<br>

---

	[1] 선형 회귀 (Linear Regression)
  
  	[2] 일반화 선형 회귀(Generalized Linear Regression, GLM)
   		[2-1] 로지스틱 회귀 (Logistic Regression) → 분류(5강)
		[2-2] 포아송 회귀 (Poisson Regression)
		[2-3] Cox의 비례위험 회귀(Cox's Proportional Hazard Regression)
     
 	[3] 다중 선형 회귀 (Multiple Linear Regression)
		[3-1] 단계적 회귀 (Stepwise Regression), 위계적 회귀 (Hierarchical Regression) 
		[3-2] 분위수 회귀 (Quantile Regression)
  
	[4] 다항 선형 회귀 (Polynomial Linear Regression)

   	[5] 정규화 (Regularized), 벌점부여 (Penalized) 선형 회귀
		[5-1] 릿지 회귀 (Ridge Regression)
		[5-2] 라쏘 회귀 (Lasso Regression)
		[5-3] 엘라스틱넷 회귀 (Elastic Net Regression)

  	[6] 비선형 회귀 (nonlinear regression)

---

	k-최근접 이웃 회귀(k-Nearest Neighbors Regression) → 분류+회귀(6강) 
	서포트 벡터 회귀 (Support Vector Regression, SVR) → 분류+회귀(6강) 
	결정 트리 회귀 (Decision Tree Regression) → 분류+회귀(6강) 
	랜덤 포레스트 회귀 (Random Forest Regression) → 분류+회귀(6강)   
 	주성분 회귀 (Principal Component Regression) → 차원축소(10강)   

---

<br>

---
# [1] 선형 회귀 (Linear Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/linear_model/index.html<br>
종속변수 y(예상값)과 독립변수(설명변수) X와의 선형 상관 관계를 모델링하는 회귀로<br>
회귀 계수(regression coefficient)를 선형 결합으로 표현할 수 있는 모델<br>
'선형'은 종속변수와 독립변수의 관계가 2차원에서는 선형으로, 3차원 공간에서는 평면으로 나타난다.<br> 

![](./images/RA.PNG)

<br>

![](./images/LinearRegression.gif)
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f
<br>

---
모델이 독립변수와 회귀계수에 대하여 선형인 경우<br>
$y = w_1x + w_0$ <br>
$y_i = β_1x_i + β_0 + ϵ_i$
###### $y_i$ : i번째 반응변수 값, $x_i$ : i번째 설명변수 값, $β_0$ : 절편 회귀계수, $β_1$ : 기울기 회귀계수, $ϵ_i$ : i번째 측정된 $y_i$의 오차 성분<br>
모든 회귀계수 각각에 대해 편미분한 결과가 다른 회귀계수를 포함하지 않는 경우에도 선형모형이라고 할 수 있다.<br>
![](./images/LRS.png)
​​
<br>

---
선형회귀는 학습을 통해 잔차 제곱들의 합인 RSS(Residual Sum of Squares)를 최소로 하는 회귀계수($W_0$과 $W_1$)를 찾는 것이 핵심.<br> 
모델 추정을 위해 예측값과 실제관측값인 잔차의 제곱(RSS)을 최소화하는 최소제곱법(OLS)을 사용한다.<br>
<img width ='500' height = '400' src = 'https://github.com/YangGuiBee/ML/blob/main/TextBook-04/images/LRd.png'>
<br>
출처 : https://blog.csdn.net/Amy_mm/article/details/79989722
<br>
비용함수(Cost function), 손실함수(loss function) =<br>
![](./images/RSSd.svg)
​<br>

**경사하강법 (Gradient Decent)**
비용 함수 f의 함숫값이 줄어드는 방향으로 함수의 계수를 일정 크기(학습량)만큼 더해나가며 f의 최솟값을 찾는 최적화 기법이다.
기울기 $Gradient(f)=∇f(x)=[ ∂f(x_0)/∂x_0, ∂f(x_1)/∂x_1,...,∂f(x_{N−1}/∂x_{N−1}]^T$
​미분 가능한 N개의 다변수 함수 f를 각 축에 대하여 편미분한 값으로, 스칼라 함수의 모든 축에 대응하는 벡터장을 생성하는 역할을 한다.
손실 함수가 조금만 복잡해져도 Global Minimum을 발견하지 못한 채 Local Minimum에 빠지기 쉽고 학습 시간이 길다.

![](./images/gradient_descent.gif)
<br>
오류가 작아지는 방향으로 w값을 보정할 수 있는 해법을 구하는 방법<br>
(1) $W_1$, $W_0$을 임의의 값으로 설정하고 첫 비용함수의 값을 계산한다.<br>
(2) $W_1$, $W_0$의 값을 주어진 횟수만큼 계속 업데이트한다.<br>
$x_{i+1} = x_i - \alpha \frac{df}{dx}(x_i)$, $x_{i+1} = x_i - \alpha \nabla f(x_i)$<br>

![](./images/w1.svg) , ![](./images/w0.svg)

---

	from sklearn.linear_model import LinearRegression

	lr = LinearRegression()
	lr.fit(train_input, train_target)

	print(lr.predict(([50]))

	# coef_ : 기울기(w1), intercept_ : 절편(w0)
	print(lr.coef_, lr.intercept_)

<br>

---
# [2] 일반화 선형 회귀(Generalized Linear Regression, GLM)
일반화 선형 회귀의 경우 선형성, 독립성, 등분산성, 정규성의 가정을 갖고 있지만, 종속변수가 연속형이 아니라면 대표적으로 오차항의 정규성 가정이 깨지게 되는데, 종속변수를 적절한 함수로 변화시킨 f(y)를 독립변수와 회귀계수의 선형결합으로 모형화한 것이다.<br>

# [2-1] 로지스틱 회귀 (Logistic Regression) → 분류(5강)

<br>

# [2-2] 포아송 회귀 (Poisson Regression)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html<br>
종속변수가 포아송 분포(Poisson Distribution)를 따르는 경우에 사용되며, 이산형 카운트 데이터를 모델링하는 데 적합하다.<br> 
포아송 분포는 단위(한정된) 시간이나 공간에서 발생하는 평균적인 사건의 횟수(λ)를 바탕으로 특정 횟수의 사건이 발생할 확률을 설명한다.<br> 
종속변수가 빈도변수로 0이상 정수이거나, 왜도가 크거나, 분포유형이 포아송 로그선형일 경우에 실시한다.<br>
참고로 평균보다 분산이 큰 경우에 적용하는 **음이항 회귀(Negative binomial regression)** 는 분산이 포아송 모델의 평균과 동일하다는 매우 제한적인 가정을 완화할 수 있다.
전통적인 음이항 회귀 모델은 포아송과 **감마(gamma regression)** 혼합 분포를 기반으로 하여 널리 사용된다.<br>
**포아송 확률변수 $X$의 확률밀도함수(probability mass function)** : $P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}$<br>
**포아송 회귀 적용 사례 :** 일정 주어진 시간 동안에 방문하는 고객의 수, 일정 주어진 생산시간 동안 발생하는 불량 수, 하룻동안 발생하는 출생자 수, 어떤 시간 동안 톨게이트를 통과하는 차량의 수, 어떤 페이지에 있는 오타의 발생률, 어떤 특정 면적의 삼림에서 자라는 소나무의 수<br>

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

# [2-3] Cox의 비례위험 회귀(Cox's Proportional Hazard Regression)
Cox의 비례위험 회귀는 생존 분석(survival analysis)에서 주로 사용되는 회귀 모델이다. 어떤 사건(event)이 일어날 때까지의 시간을 대상으로 분석하는 통계방법으로 사건과 사건 사이의 예측 회귀 모형을 분석한다. 이 모델은 사건(예: 사망, 질병 발병, 기계 고장 등)이 발생할 때까지의 시간과 그 사건이 발생할 확률(위험율) 사이의 관계를 설명한다. 주어진 독립변수 값에 대해 위험율($hazard ratio(log(h(t)/h_0(t)))$)이 시간에 걸쳐 일정한 비율로 유지(두 피험자에 대해 위험율의 비율이 시간이 지나도 일정하게 유지)된다고 가정한다. 위험율(HR)이 1보다 크면 위험이 증가하고, 1보다 작으면 위험이 감소하는 것으로 평가한다. 환자가 특정 치료 후 생존할 확률을 예측, 기계 부품이 고장날 때까지의 시간을 분석, 사회학 연구에서 결혼생활이 파탄날 확률을 예측할 때 활용한다. 만약 비례 위험 가정이 만족되지 않으면 Cox 회귀 모델의 결과가 왜곡될 수 있으므로, 이 경우에는 비례 위험 가정을 검토하거나 시간을 고려한 상호작용 변수를 추가해야 한다.<br>
<br>

| 구분   | 포아송 회귀                                  | Cox 회귀                                                   |
|--------|----------------------------------------------|------------------------------------------------------------|
| 목적   | 사건 발생 횟수 예측                          | 사건이 발생할 때까지의 시간과 그 사건의 위험율을 분석      |
| 데이터 | 주로 이산형(정수)                            | 생존시간과 같은 연속형                                     |
| 가정   | 포아송 분포와 로그 링크 함수                 | 비례위험                                                   |
| 사례   | 범죄율, 질병 발생률 등 사건 발생 횟수의 예측 | 환자의 생존율, 부품의 고장 시간 등 생존 분석과 관련된 문제 |

<br>

---
# [3] 다중회귀 (Multiple Regression)
독립변수 X가 2개 이상인 회귀<br>
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

# [3-1] 단계적 회귀 (Stepwise Regression), 위계적 회귀 (Hierarchical Regression) 
여러 독립변수 중에서 종속변수를 가장 잘 설명하는 변수들을 선택하는 방법<br>
**단계적 회귀 (Stepwise Regression)** 는 독립 변수들을 자동으로 모델에 추가하거나 제거하여 최적의 모델을 탐색(변수의 추가나 제거가 통계적으로 유의미한지 여부에 따라 이루어짐)<br>
예를 들어, 변수를 추가할 때마다 F 통계량이유의미하게 증가하는지 확인하거나, 제거할 때마다 변수의 t 통계량이 유의미하게 감소하는지 확인<br> 
장점: 자동으로 변수를 선택하므로 모델이 데이터에 더 잘 맞을 가능성이 있음<br>
**위계적 회귀 (Hierarchical Regression)** 는 독립 변수들을 미리 정의한 순서에 따라 모델에 추가하는 것으로,<br>
이론적으로 중요한 변수부터 시작하여 덜 중요한 변수를 차례로 추가하는 방식<br>
장점: 이론적 근거에 따라 변수를 추가하므로 결과 해석이 이론적으로 타당함.<br>

<br>

# [3-2] 분위수 회귀 (Quantile Regression)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html<br>
반응 변수의 조건부 분위수를 모델링 할때 사용되는 선형 회귀의 확장 버전<br>
1) 선형 회귀 조건이 충족되지 않는 경우<br>
2) 오차의 분산이 큰 경우<br>
3) Robust한 결과를 위하여<br>
4) 많은 이상치의 영향을 줄이기 위하여<br>
5) 점 추정이 아닌 구간추정을 통해 결과의 정확도를 높이기 위하여<br>
6) 반응변수의 스프레드를 같이 살펴보기 위하여<br>
7) 회귀곡선에 대한 설득력을 높이기 위하여<br>

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

---
# [4] 다항 회귀 (Polynomial Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions<br>
직선이 아닌 곡선 형태의 관계의 경우, 독립변수에 제곱이나 로그(log) 등을 취해 보면서 실시하는 모델링<br>
$y = w_1x_1 + w_2x_2^2 + ... + w_nx_n^n + w_0$ <br>
<br>
![](./images/PolynomialFeatures.png)

<br>
편향이 높으면 분산은 낮아짐 : 과소적합(Under fitting), 분산이 높으면 편향이 낮아짐 : 과대적합(Over fitting)<br>
  
![](./images/ddd.PNG)



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


	입력데이터의 Feature들이 너무 많은 경우(Feature수에 비해 관측치 수가 적은 경우) 과적합이 발생
	→ 
	(해결방안1) 데이터를 더 수집하거나 불필요한 Features들을 제거
	(해결방안2) 가중치(회귀계수)에 페널티 값을 적용하는 규제(Regularization)를 통해 
 	            Feature들에 곱해지는 가중치가 커지지 않도록 제한

<br>

---
# [5] 정규화 (Regularized), 벌점부여 (Penalized) 선형 회귀
규제(Regularization) : 비용함수에 alpha값으로 패널티를 부여해서 회귀계수값의 크기를 감소시켜서 과적합을 개선<br>
비용함수의 목표 = $Min(RSS(W) + alpha * ||W||_2^2)$

# [5-1] 릿지 회귀 (Ridge Regression)
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

# [5-2] 라쏘 회귀 (Lasso Regression)
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

# [5-3] 엘라스틱넷 회귀 (Elastic Net Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html<br>
릿지와 라쏘의 결합으로 L1규제로 Feature 수를 줄임과 동시에 L2규제로 계수값의 크기를 조정하는 패널티를 부과하여 회귀모델을 생성<br>

	from sklearn.linear_model import ElasticNet
	elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

<br>

![](./images/L1L2.png)
<br>
출처 : https://savannahar68.medium.com/getting-started-with-regression-a39aca03b75f

<br>

---
# [6] 비선형 회귀 (nonlinear regression)
데이터를 어떻게 변형하더라도 파라미터를 선형 결합식으로 표현할 수 없는 모델로 회귀모형에 주어진 회귀식이 모수들의 비선형함수로 나타나는 경우 선형회귀에서 회귀계수는 설명변수의 변화량에 따른 반응변수의 평균변화량으로 해석되지만, 비선형회귀에서는 각 모수가 특정한 의미를 가지게 된다.<br>
(1) 다항 회귀 (Polynomial Regression)
 $y = β_0 + β_1X + β_2X^2 +⋯+ β_nX^n + ϵ$

(2) 지수 회귀 (Exponential Regression)
 $y = αe^{βX} + ϵ$ 
 $ln(y) = ln(α) + βX + ϵ$

(3) 로그 회귀 (Logarithmic Regression)
 $y = α + βln(X) + ϵ$
 $∂y/∂x = β/x$

(4) 다중 회귀 (Multiple Regression)
 $y = α + β_1X_1 + β_2X_2^2 + β_3sin(X_3) + ϵ$

(5) 시그모이드 회귀 (Sigmoid Regression)
 $y = 1/(1+e^−{βX}) + ϵ$
 $ln(y/(1−y)) = βX + ϵ$

(6) 전력 회귀 (Power Regression)
 $y = αx^β + ϵ$
 $∂y/∂x = α⋅β⋅x^{β−1}$

(7) 포아송 회귀 (Poisson Regression)
 $ln(y) = α + βX + ϵ$

(8) 감마 회귀 (Gamma Regression)
 $y = αX^β + ϵ$
 $ln(y) = ln(α) + βln(X) + ϵ$

(9) 베이즈 회귀 (Bayesian Regression)
 $y = β_0 + β_1X_1 +⋯+ β_nX_n + ϵ$

(10) 스플라인 회귀 (Spline Regression)
 $y = β_iB_i(X)의 합 + ϵ$

(11) 로버스트 회귀 (Robust Regression)
 $y = β_0 + β_1X_1 +⋯+ β_nX_n + ϵ$

(12) 커널 회귀 (Kernel Regression)
 $y = α_iK(X,X_i)의 합 + ϵ$

(13) 구형 회귀 (Quadratic Regression)
 $y = β_0 + β_1x + β_2x^2 + ϵ$
 $∂y/∂x = β_1 + 2β_2x$

<br>


