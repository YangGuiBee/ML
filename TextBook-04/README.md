#  04 : 지도 학습 (Supervised Learning, SL) : 회귀 (regression)

---

	[1] 선형 회귀 (Linear Regression)
	[2] 다항 회귀 (Polynomial Regression)
	[3] 릿지 회귀 (Ridge Regression)
	[4] 라쏘 회귀 (Lasso Regression)
	[5] 엘라스틱넷 회귀 (Elastic Net Regression)
	  
---

# [1] 선형 회귀 (Linear Regression)
종속변수 y와 독립변수(설명변수) X와의 선형 상관 관계를 모델링하는 회귀<br>
특히 특히 독립변수(설명변수) X가 2개 이상인 경우는 다중선형회귀 (Multiple  Linear Regression)

<br>

# [2] 다항 회귀 (Polynomial Regression)
데이터 표현이 선형이 아닌 곡선으로 나타나는 경우에 사용하는 회귀

<br>

# [3] 릿지 회귀 (Ridge Regression)
모델의 설명력에 기여하지 못하는 독립변수의 회귀계수 크기를 0에 근접하도록 축소시키는 분석<br>
L2-norm 페널티항을 통해 일반 선형회귀 모델에 페널티를 부과하는 방법으로 회귀계수를 축소

<br>

# [4] 라쏘 회귀 (Lasso Regression)
릿지회귀모델과 다르게 설명력에 기여하지 못하는 독립변수의 회귀계수를 0으로 만드는 분석<br>
L1-norm 패널티항으로 회귀모델에 패널티를 부과함으로써 회귀계수를 축소

<br>

# [5] 엘라스틱넷 회귀 (Elastic Net Regression)
릿지와 라쏘의 결합<br>
L1-norm 과 L2-norm을 모두 이용하여 패널티를 부과하여 회귀모델을 생성<br>

<br>


