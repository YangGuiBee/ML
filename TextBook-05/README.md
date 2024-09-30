#  05 : 지도 학습 (Supervised Learning, SL) : 분류 (classification)


---

	[1] 경사하강법 기반 알고리즘 : Logistic Regression

	[2] 확률 기반 알고리즘 : Naive Bayes Classifier

	[3] 거리 기반 알고리즘 
		[3-1] K-Nearest Neighbor Classification → 6강
		[3-2] Support Vector Machine Classification → 6강

	[4] 트리기반 알고리즘
		[4-1] Decision Tree Classification → 6강
		[4-2] Random Forest Classification → 6강
		[4-3] 앙상블 기반 :  AdaBoost, Gradient Boosting Tree (GBT), lightGBM, XGBoost, CatBoost → 12강
  
---

	k-최근접 이웃 분류(k-Nearest Neighbors Classification) → 분류+회귀
	서포트 벡터 분류 (Support Vector Classification, SVC) → 분류+회귀
	결정 트리 분류 (Decision Tree Classification) → 분류+회귀
	랜덤 포레스트 분류 (Random Forest Classification) → 분류+회귀

---

![](./images/SLC.png)
<br>출처 : https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501

<br>

# [1] 로지스틱 회귀 (Logistic Regression)
▣ 가이드 : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html<br>
이름에 Regression이 들어가지만 Sigmoid 함수를 활용한 이진분류에 해당. 선형 회귀는 특정 예측 값을 반환하지만, sigmoid 함수를 활용하면 특정 값을 기점으로 0 또는 1 값을 반환하는 분류 모델이라고 할 수 있다.<br>

▣ 응용분야
 - 제조 : 로지스틱 회귀 분석을 사용하여 기계류의 부품 고장 확률을 추정한 다음 이 추정치를 기반으로 유지 보수 일정을 계획하여 향후 고장 발생을 최소화<br>
 - 의료 : 의학 연구원들은 환자의 질병 발생 가능성을 예측하여 예방 진료와 치료를 계획하는데 이때 가족력이나 유전자가 질병에 미치는 영향을 비교하는 데 로지스틱 회귀 모델을 사용<br> 
 - 금융 : 금융 거래에서 사기 행위를 분석하고 대출 신청 및 보험 신청 건의 위험도를 평가해야 하는데 이러한 문제에는 고위험이거나 저위험이거나, 사기이거나 사기가 아닌 것과 같은 명확한 결과가 있기 때문에 로지스틱 회귀 모델에 적합<br>  
 - 마케팅 : 온라인 광고 도구는 로지스틱 회귀 모델을 사용하여 사용자가 광고를 클릭할지 여부를 예측한 결과를 활용하여 마케터는 다양한 단어와 이미지에 대한 사용자의 반응을 분석하고 고객이 관심을 가질 만한 효과적인 광고 제작<br>


<br>

# [2] 나이브 베이즈 (Naive Bayes)

<br>



