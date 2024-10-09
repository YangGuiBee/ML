#  12 : 비지도 학습 (Unsupervised Learning, UL) : 강화학습, 앙상블학습

---

## 강화 학습 (Reinforcement Learning, RL)
<br>

    [1] Q-learning
    [2] Deep Q-Network (DQN)
    [3] State-Action-Reward-State-Action (SARSA)
    [4] 유전 알고리즘 (Genetic Algorithm)
    [5] Asynchronous Advantage Actor-Critic (A3C)
  

## 앙상블 학습 (Ensemble Learning, EL)
<br>

    [6] 스태킹 (Stacking)
    [7] 배깅 (Bagging)
    [8] 부스팅 (Boosting)

---  
# 강화 학습 (Reinforcement Learning, RL)

# [1] Q-learning

<br>

# [2] Deep Q-Network (DQN)

<br>

# [3] State-Action-Reward-State-Action (SARSA)

<br>

# [4] 유전 알고리즘 (Genetic Algorithm)

<br>

# [5] Asynchronous Advantage Actor-Critic (A3C)

<br>

---

# 앙상블 학습 (Ensemble Learning, EL)
▣ API : https://scikit-learn.org/stable/api/sklearn.ensemble.html<br>
▣ 정의 : 앙상블 학습이란 다수의 기초 알고리즘(base algorithm)을 결합하여 더 나은 성능의 예측 모델을 형성하는 것을 말하며, 사용 목적에 따라 배깅(Bagging), 부스팅(Boosting), 스택킹(Stacking)으로 분류된다.<br>

# [6] 스태킹 (Stacking)
▣ 정의 : 스태킹은 서로 다른 종류의 기반 모델(base model) 여러 개를 학습한 후, 이들의 예측 결과를 결합하는 방식이다. 개별 모델의 예측 결과를 다시 하나의 메타 모델(meta-model)로 학습시켜 최종 예측을 수행한다.<br>
▣ 필요성 : 단일 모델의 약점을 보완하기 위해 서로 다른 유형의 모델을 조합함으로써 더 나은 성능을 도출할 수 있다. 예를 들어, 결정 트리, 서포트벡터머신(SVM), 신경망 등 다양한 모델을 결합할 수 있다.<br>
▣ 장점 : 서로 다른 모델의 장점을 결합하여 더욱 강력한 예측 성능을 낼 수 있으며, 다양한 모델의 편향과 분산을 보완할 수 있다.<br>
▣ 단점 : 모델 조합이 복잡해질수록 계산 비용이 커지고, 메타 모델을 학습하는 데 추가적인 시간이 소요되며 과적합(overfitting)의 위험이 있다.<br>
▣ 응용분야 : 여러 모델의 특성이 유용할 때 사용한다. 예를 들어, 금융 예측, 이미지 분류 등 다양한 문제에서 활용된다.<br>
▣ 모델식 : $𝑓_1$ 은 각각의 개별 모델, $𝑓_2$ 는 메타 모델, $\widehat{y}=f_2(f_1(x_1),f_1(x_2),...f_1(x_n))$<br>
▣ python 예제 : 

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris

    # 데이터 로드 및 분할
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 개별 모델 정의
    estimators = [('svc', SVC(probability=True)),('tree', DecisionTreeClassifier())]

    # 스태킹 모델 정의
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking_clf.fit(X_train, y_train)

    # 성능 평가
    print(stacking_clf.score(X_test, y_test))
    
<br>

# [7] 배깅 (Bagging)
▣ 정의 : 배깅은 동일한 모델을 여러 번 학습하되, 각 학습마다 다른 데이터 샘플을 사용한다. 주로 부트스트랩(bootstrap) 방법으로 샘플링된 데이터로 모델을 학습하며, 최종 예측은 개별 모델의 예측 결과를 평균 또는 투표로 결합한다. 대표적인 알고리즘은 랜덤 포레스트(Random Forest)이다.<br>
▣ 필요성 : 단일 모델이 데이터의 특정 부분에 과적합하는 것을 방지하고, 예측의 안정성을 높이기 위해 사용된다.<br>
▣ 장점 : 분산을 줄여 예측 성능을 향상시키며, 과적합(overfitting)을 방지하는 데 도움이 된다.<br>
▣ 단점 : 편향을 줄이는 데는 효과적이지 않을 수 있으며, 많은 모델을 학습하므로 계산 자원이 많이 필요하다.<br>
▣ 응용분야 : 랜덤 포레스트처럼 결정 트리 기반 모델에서 이미지 분류, 텍스트 분류, 금융 예측 등에 널리 사용된다.
▣ 모델식 : 
▣ python 예제 : 

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # 데이터 로드 및 분할
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 배깅 모델 정의
    bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    bagging_clf.fit(X_train, y_train)

    # 성능 평가
    print(bagging_clf.score(X_test, y_test))

<br>

# [8] 부스팅 (Boosting)
▣ 정의 : 부스팅은 약한 학습기(weak learner)를 연속적으로 학습시키며, 이전 학습에서 잘못 예측한 데이터에 가중치를 부여하여 다음 모델이 이를 더 잘 학습할 수 있도록 한다. 대표적인 알고리즘으로는 AdaBoost, Gradient Boosting, XGBoost 등이 있다.<br>
▣ 필요성 : 약한 학습기를 여러 번 반복하여 강력한 학습기를 만들 수 있으며, 특히 잘못된 예측에 집중하여 성능을 점진적으로 개선한다.<br>
▣ 장점 : 모델이 연속적으로 개선되기 때문에 높은 예측 성능을 보일 수 있으며, 오류를 줄이는 데 매우 효과적이다.<br>
▣ 단점 : 연속적인 학습 과정에서 모델이 과적합할 위험이 있으며, 학습 속도가 느릴 수 있다.<br>
▣ 응용분야 : 금융 예측, 분류 문제, 회귀 분석 등에서 많이 사용되며, 특히 XGBoost는 대회에서 많이 사용된다.<br>
▣ 모델식 : 
▣ python 예제 : 

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # 데이터 로드 및 분할
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 부스팅 모델 정의
    boosting_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    boosting_clf.fit(X_train, y_train)

    # 성능 평가
    print(boosting_clf.score(X_test, y_test))

<br>

