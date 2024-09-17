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

# [2] 평균 제곱 오차 (Mean Squared Error, MSE)
선형 회귀 모델에서 예측값과 실제값의 차이를 제곱한 것들의 평균<br>
$MSE = (1/(n-k-1)) * ∑(i=1 to n) (yi - ŷi)^2$ <br>
$SSE = ∑(i=1 to n) (yi - ŷi)^2$ <br>
$MSE = SSE / (n-K-1)$ <br>
(n: 샘플 데이터의 개수, k: 독립변수의 개수, yi: 실제값, ŷi: 모델의 예측값, SSE: Sum of Squared Errors)<br>
MSE가 작을수록 모델의 예측 성능이 높다고 해석

<br>

# [3] 평균 제곱 오차(로그적용) (Mean Squared Log Error, MSLE)

# [4] 평균 제곱근 오차 (Root Mean Squared Error, RMSE)

# [5] 평균 제곱근 오차(로그적용) (Root Mean Squared Log Error, RMSLE)

# [6] 평균 절대 비율 오차 (Mean Absolute Percentage Error, MAPE)

# [7] 평균 절대 비율 오차(절대값제외) (Mean Percentage Error, MPE)

# [8] R2 score
R-squared는 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 지표<bbr>
R-squared는 0에서 1 사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미<br>
$R-Squared = SSR / SST = (SST – SSE) / SST = 1 – (SSE / SST)$ <br>
$SSR = ∑(i=1 to n) (ŷi - ȳ)^2$ <br>
$SST = ∑(i=1 to n) (yi - ȳ)^2$ <br>

$SST = ∑(yi - ȳ)²$ <br>
$= ∑[(yi - ŷi) + (ŷi - ȳ)]²$ <br>
$= ∑(yi - ŷi)² + 2∑(yi - ŷi)(ŷi - ȳ) + ∑(ŷi - ȳ)²$ <br>
$= SSE + 2∑(yi - ŷi)(ŷi - ȳ) + SSR$ <br>
$= SSE + SSR (단, 2∑(yi - ŷi)(ŷi - ȳ)는 0)$ <br>

<br>

![](./images/SST.png)
<br>
출처 : https://medium.com/coders-mojo/data-science-and-machine-learning-projects-mega-compilation-part-5-e50baa2faa85
