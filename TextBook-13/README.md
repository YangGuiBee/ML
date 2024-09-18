#  13 : 회귀 평가 지표

---

	[1] 평균 오차 (Mean Error, ME)
 	[2] 평균 절대 오차 (Mean Absolute Error, MAE)
	[3] 평균 제곱 오차 (Mean Squared Error, MSE)
	[4] 평균 제곱 오차(로그적용) (Mean Squared Log Error, MSLE)
	[5] 평균 제곱근 오차 (Root Mean Squared Error, RMSE)
	[6] 평균 제곱근 오차(로그적용) (Root Mean Squared Log Error, RMSLE)
 	[7] 평균 비율 오차 (Mean Percentage Error, MPE)
	[8] 평균 절대 비율 오차 (Mean Absolute Percentage Error, MAPE)
	[9] 평균 절대 규모 오차 (MASE(Mean Absolute Scaled Error)
	[10] R2 score
	  
---

# [1] 평균 절대 오차 (Mean Error, ME)
![](./images/ME.svg)
<br>
예측오차의 산술평균을 의미<br>

	def ME(y, t):
		return (y-t).mean(axis=None)

<br>

# [2] 평균 절대 오차 (Mean Absolute Error, MAE)
![](./images/MAE.svg)
<br>
실제 정답 값과 예측 값의 차이를 절댓값으로 변환한 뒤 합산하여 평균을 구한다.<br>
특이값이 많은 경우에 주로 사용된다.<br>
값이 낮을수록 좋다.<br>
직관적이고 정답 및 예측 값과 같은 단위를 가지고, MSE, RMSE에 비해, 오차값이 outlier의 영향을 상대적으로 적게 받는 장점<br>
절댓값을 취하므로 underestimates/overestimates인지에 대한 판단을 할 수 없으며, 스케일 의존적(scal dependency)으로 모델마다 에러 크기가 동일해도 에러율은 동일하지 않은 단점<br>

	def MAE(y, t):
    	return (abs(y - t)).mean(axis=None)
   
<br>

# [3] 평균 제곱 오차 (Mean Squared Error, MSE)
![](./images/MSE.svg)
<br>
실제 정답 값과 예측 값의 차이를 제곱(예측값과 실제값 차이의 면적)한 뒤 평균을 구한다.<br>
값이 낮을수록 좋다.<br>
직관적인 장점<br>
예측 변수와 단위가 다르며, 오차를 제곱하기 때문에 이상치에 민감(제곱하기 때문에 1미만의 에러는 작아지고 그 이상의 에러는 커짐), 제곱을 씌우게 되어 underestimates/overestimates인지 파악하기 힘들며, 스케일 의존적(scal dependency)이라 모델마다 에러러 크기가 동일해도 에러율은 동일하지 않은 단점<br>
CF : 오차제곱합(SSE)와 유사하지만 오차제곱합으로는 실제 오차가 커서 값이 커지는 것인지 데이터의 양이 많아서 값이 커지는 것인지를 구분할 수 없게 된다.<br>

	def MSE(y, t):
    	return ((y-t)**2).mean(axis=None)

	def SSE(y, t):
    	return 0.5*np.sum((y-t)**2)

<br>

# [4] 평균 제곱 오차(로그적용) (Mean Squared Log Error, MSLE)
![](./images/MSLE.svg)
<br>
MSE에 로그를 적용한 것이다.<br> 
결정 값이 클수록 오류값도 커지기 때문에 일부 큰 오류값들로 인해 전체 오류값이 커지는 것을 막아준다.<br>

	def MSLE(y, t):
		return np.log((y-t)**2).mean(axis=None)

<br>

# [5] 평균 제곱근 오차 (Root Mean Squared Error, RMSE)
![](./images/RMSE.svg)
<br>
MSE에 루트는 씌워서 에러를 제곱해서 생기는 값의 왜곡이 줄어든다.<br>
값이 낮을수록 좋다.<br>
직관적인 장점<br>
제곱 후 루트를 씌우기 때문에 MAE처럼 실제 값에 대해 underestimates/overestimates인지 파악하기 힘들고, 스케일 의존적(scal dependency)으로 모델마다 에러 크기가 동일해도 에러율은 동일하지 않은 단점<br>

	def RMSE(y, t):
		return np.sqrt(((y - t) ** 2).mean(axis=None))

<br>

# [6] 평균 제곱근 오차(로그적용) (Root Mean Squared Log Error, RMSLE)
![](./images/RMSLE.svg)
<br>
RMSE값에 로그를 취한 값이다.<br>
결정 값이 클 수록 오류 값도 커지기 때문에 일부 큰 오류 값들로인해 전체 오류값이 커지는 것을 막아준다.<br>

	def RMSLE(y, t):
		return np.log(np.sqrt(((y - t) ** 2).mean(axis=None)))

<br>

# [7] 평균 비율 오차 (Mean Percentage Error, MPE)
![](./images/MPE.svg)
<br>
절대적인 의미의 예측오차뿐 아니라 상대적인 의미의 예측오차가 필요할 경우에 계산한다.<br>
음수면 overperformance, 양수면 underperformance으로 판단<br>
모델이 underestimates/overestimates인지 판단할 수 있다는 장점<br>

	def MPE(y, t):
		return (((y-t)/y)*100).mean(axis=None)

<br>

# [8] 평균 절대 비율 오차 (Mean Absolute Percentage Error, MAPE)
![](./images/MAPE.svg)
<br>
MAE를 비율, 퍼센트로 표현하여 스케인 의존적 에러의 문제점을 개선한다.<br>
값이 낮을수록 좋다.<br>
직관적이고, 다른 모델과 에러율 비교가 쉬운 장점<br>
실제 정답보다 낮게 예측했는지, 높게 했는지를 파악하기 힘들고 실제 정답이 1보다작을 경우,무한대의 값으로 수렴할 수 있는 단점<br>

	def MAPE(y, t):
		return ((abs((y-t)/y))*100).mean(axis=None)

<br>

# [9] 평균 절대 규모 오차 (MASE(Mean Absolute Scaled Error)
![](./images/MASE.svg) ![](./images/MASE1.svg)
<br>
데이터를 척도화하여 이를 기준으로 예측오차의 절대값에 대한 평균을 낸 값<br>
스케일에 대한 의존성이 낮다는 장점<br>

	def MASE(y, t):
		n = len(y)
		d = np.abs(np.diff(y)).sum() / (n - 1)
		errors = abs(y-t)
		return errors.mean(axis=None)/d

<br>

# [10] R2 score
![](./images/SSR.svg)
<br>
![](./images/SST.svg)
<br>
![](./images/R.svg)
<br>
다른 지표(MAE, MSE, RMSE)들은 모델마다 값이 다르기 때문에 절대 값만 보고 선능을 판단하기 어려운 반면, $R^2$ score는 상대적인 성능을 나타내기 비교가 쉽다.<br>
실제 값의 분산 대비 예측값의 분산 비율을 의미한다.<br>
0에서 1 사이의 값을 가지며, 1에 가까울 수록 좋다.<br>

<br>

출처 : https://aliencoder.tistory.com/43

<br>

![](./images/SST.png)
<br>
출처 : https://medium.com/coders-mojo/data-science-and-machine-learning-projects-mega-compilation-part-5-e50baa2faa85
