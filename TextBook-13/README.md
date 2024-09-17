#  13 : 회귀 평가 지표

---

	[1] 평균 절대 오차 (Mean Absolute Error, MAE)
	[2] 평균 제곱 오차 (Mean Squared Error, MSE)
	[3] 평균 제곱 오차(로그적용) (Mean Squared Log Error, MSLE)
	[4] 평균 제곱근 오차 (Root Mean Squared Error, RMSE)
	[5] 평균 제곱근 오차(로그적용) (Root Mean Squared Log Error, RMSLE)
	[6] 평균 절대 비율 오차 (Mean Absolute Percentage Error, MAPE)
	[7] 평균 절대 비율 오차(절대값제외) (Mean Percentage Error, MPE)
	[8] R2 score

	  
---

# [1] 평균 절대 오차 (Mean Absolute Error, MAE)
![](./images/MAE.svg)
<br>
실제 정답 값과 예측 값의 차이를 절댓값으로 변환한 뒤 합산하여 평균을 구한다.<br>
특이값이 많은 경우에 주로 사용된다.<br>
값이 낮을수록 좋다.<br>
직관적이고 정답 및 예측 값과 같은 단위를 가지는 장점<br>
실제 정답보다 낮게 예측했는지, 높게 했는지를 파악하기 힘들고, 스케일 의존적(scal dependency)으로 모델마다 에류 크기가 동일해도 에러율은 동일하지 않은 단점<br>

<br>

# [2] 평균 제곱 오차 (Mean Squared Error, MSE)
![](./images/MSE.svg)
<br>
실제 정답 값과 예측 값의 차이를 제곱한 뒤 평균을 구한다.<br>
값이 낮을수록 좋다.<br>
직관적인 장점<br>
제곱하기 때문에 1미만의 에러는 작아지고, 그 이상의 에러는 커지고, 실제 정답보다 낮게 예측했는지, 높게 했는지를 파악하기 힘들고, 스케일 의존적(scal dependency)이라 모델마다 에류 크기가 동일해도 에러율은 동일하지 않은 단점<br>

<br>

# [3] 평균 제곱 오차(로그적용) (Mean Squared Log Error, MSLE)

# [4] 평균 제곱근 오차 (Root Mean Squared Error, RMSE)
![](./images/RMSE.svg)
<br>
MSE에 루트는 씌워서 에러를 제곱해서 생기는 값의 왜곡이 줄어든다.<br>
값이 낮을수록 좋다.<br>
직관적인 장점<br>
제곱하기 때문에 1미만의 에러는 작아지고, 그 이상의 에러는 커지고. 실제 정답보다 낮게 예측했는지, 높게 했는지를 파악하기 힘들고, 스케일 의존적(scal dependency)으로 모델마다 에류 크기가 동일해도 에러율은 동일하지 않은 단점<br>

<br>

# [5] 평균 제곱근 오차(로그적용) (Root Mean Squared Log Error, RMSLE)

# [6] 평균 절대 비율 오차 (Mean Absolute Percentage Error, MAPE)
![](./images/MAPE.svg)
<br>
MAE를 비율, 퍼센트로 표현하여 스케인 의존적 에러의 문제점을 개선한다.<br>
값이 낮을수록 좋다.<br>
직관적이고, 다른 모델과 에러율 비교가 쉬운 장점<br>
실제 정답보다 낮게 예측했는지, 높게 했는지를 파악하기 힘들고 실제 정답이 1보다작을 경우,무한대의 값으로 수렴할 수 있는 단점<br>

<br>

# [7] 평균 절대 비율 오차(절대값제외) (Mean Percentage Error, MPE)
![](./images/MPE.svg)
<br>
MAPE에서 절대값을 제외하여 계산한다.<br>
음수면 overperformance, 양수면 underperformance으로 판단<br>

# [8] R2 score

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

![](./images/SST.png)
<br>
출처 : https://medium.com/coders-mojo/data-science-and-machine-learning-projects-mega-compilation-part-5-e50baa2faa85
