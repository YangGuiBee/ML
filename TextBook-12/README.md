#  12 : 비지도 학습(Unsupervised Learning, UL) : 강화학습, 앙상블학습

---

## 강화 학습(Reinforcement Learning, RL)
<br>

    [1] Q-learning
    [2] Deep Q-Network(DQN)
    [3] State-Action-Reward-State-Action(SARSA)
    [4] 유전 알고리즘(Genetic Algorithm)
    [5] Asynchronous Advantage Actor-Critic(A3C)
  

## 앙상블 학습(Ensemble Learning, EL)
<br>

    [6] 스태킹(Stacking)
    [7] 배깅(Bagging)
    [8] 부스팅(Boosting)

---  
# 강화 학습(Reinforcement Learning, RL)

# [1] Q-learning
▣ 정의 : Q-learning은 값 기반 강화 학습의 대표적인 알고리즘으로, 상태-행동 쌍에 대한 Q값을 학습해 최적의 정책을 찾는 방법이다. 상태에서 어떤 행동을 선택할지 결정하는 Q함수를 학습하며, 최적 정책을 따르기 위해 Q값을 최대화하는 방향으로 행동한다.<br>
▣ 필요성 : 모델에 대한 사전 지식 없이 환경 내에서 에이전트가 스스로 학습할 수 있는 능력을 제공하며, 상태 공간이 클 때도 적합하게 사용할 수 있다.<br>
▣ 장점 : 모델 프리 방식이라 환경의 동작을 미리 알 필요가 없으며, 수렴할 경우 최적의 정책을 보장한다.<br>
▣ 단점 : 상태 공간이 매우 크거나 연속적인 경우, Q-table이 메모리와 시간 측면에서 비효율적일 수 있으며, 학습 속도가 느리고, 보상이 주기적으로만 주어지는 경우 최적의 정책을 찾기 어려울 수 있다.<br>
▣ 응용분야 : 게임 플레이, 로봇 제어, 자율 주행, 네트워크 트래픽 제어 등<br>
▣ 모델식 : Q-learning 업데이트식으로 Q(s,a)는 상태 𝑠에서 행동 𝑎를 선택할 때의 Q값, α는 학습률, 𝛾는 할인 계수,𝑟은 현재 보상, max_𝑎′𝑄(𝑠′,𝑎′)는 다음 상태 𝑠′ 에서 가능한 최대 Q값.
▣ 주요 알고리즘 : Q값을 0으로 초기화후 현재 상태에서 가능한 행동을 선택 (탐험/탐색 균형), 보상을 받고 다음 상태로 이동, Q값을 업데이트. 종료 상태에 도달할 때까지 반복.
▣ python 예제 : 

    import numpy as np

    # 환경 설정 (간단한 그리드 월드 환경 가정)
    n_states = 5
    n_actions = 2
    Q = np.zeros((n_states, n_actions))

    alpha = 0.1  # 학습률
    gamma = 0.9  # 할인 계수
    epsilon = 0.1  # 탐험 확률

    def choose_action(state):
        if np.random.uniform(0, 1) < epsilon:
           return np.random.choice(n_actions)
        else:
           return np.argmax(Q[state, :])

    def update_q(state, action, reward, next_state):
        predict = Q[state, action]
        target = reward + gamma * np.max(Q[next_state, :])
        Q[state, action] = predict + alpha * (target - predict)

    # 예시 학습 반복
    for episode in range(100):
        state = np.random.randint(0, n_states)
        while state != 4:  # 종료 상태 가정
          action = choose_action(state)
          next_state = np.random.randint(0, n_states)
          reward = 1 if next_state == 4 else 0
          update_q(state, action, reward, next_state)
          state = next_state
    print(Q)
    
<br>

# [2] Deep Q-Network(DQN)
▣ 정의 : DQN은 Q-learning을 딥러닝에 결합한 알고리즘으로, Q-table 대신 심층 신경망을 사용해 Q값을 근사하며, 주로 상태 공간이 매우 크거나 연속적인 문제에서 사용된다.<br>
▣ 필요성 : Q-table을 사용할 수 없는 고차원 환경에서 Q-learning을 효과적으로 적용하기 위해 신경망을 사용하여 Q값을 근사한다.<br>
▣ 장점 : 고차원 연속 상태 공간에서 사용 가능하며, 경험 재플레이(experience replay)와 타깃 네트워크로 학습 안정성을 높일 수 있다.<br>
▣ 단점 : 신경망 학습으로 인해 높은 계산 비용이 필요하며, 과적합 위험이 있으며, 잘못 설정된 하이퍼파라미터로 인해 학습이 불안정해질 수 있다.<br>
▣ 응용분야 : 비디오 게임(예: Atari 게임), 로봇 제어, 자율 주행 등.<br>
▣ 모델식 : DQN에서 신경망을 사용한 Q-learning 업데이트 θ는 현재 신경망의 가중치,𝜃′ 는 타깃 신경망의 가중치.<br>
▣ 주요 알고리즘 : 신경망 초기화 및  경험 재플레이 메모리 초기화. 현재 상태에서 행동 선택 (탐험/탐색 균형). 경험을 메모리에 저장. 일정 주기마다 경험 샘플을 이용해 신경망을 업데이트. 타깃 네트워크 주기적으로 업데이트를 반복<br>
▣ python 예제 : 

    import numpy as np
    import tensorflow as tf
    from collections import deque

    n_states = 5
    n_actions = 2

    # 신경망 모델 정의
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(24, input_dim=n_states, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(n_actions, activation='linear')])
    model.compile(optimizer='adam', loss='mse')

    # Q-learning 파라미터 설정
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    memory = deque(maxlen=2000)

    def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(n_actions)
    state = np.reshape(state, [1, n_states])
    return np.argmax(model.predict(state))

    def replay():
        global epsilon
        if len(memory) < batch_size: return
        minibatch = np.random.choice(len(memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done: target = reward + gamma * np.amax(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
        if epsilon > epsilon_min: epsilon *= epsilon_decay

    # 학습 반복 (예시)
    for episode in range(1000):
        state = np.random.rand(n_states)
        done = False
        while not done:
            action = choose_action(state)
            next_state = np.random.rand(n_states)
            reward = 1 if np.random.rand() > 0.5 else 0
            done = True if reward == 1 else False
            memory.append((state, action, reward, next_state, done))
            replay()
    print("학습 완료")

<br>

# [3] State-Action-Reward-State-Action(SARSA)
▣ 정의 : SARSA는 강화 학습의 한 기법으로, 상태-행동-보상-다음 상태-다음 행동(State-Action-Reward-State-Action)의 연속적인 관계에서 학습하는 방법이다. Q-learning과 달리 SARSA는 에이전트가 선택한 행동을 기반으로 학습하며 에이전트가 현재 행동과 다음 행동을 통해 학습하는 on-policy 방법이다.<br>
▣ 필요성 : 정책을 미리 고정한 상태에서 Q-learning처럼 탐험과 학습을 분리하지 않고, 정책을 유지하며 학습할 때 유리하다. SARSA는 실제로 에이전트가 수행하는 행동을 기반으로 학습하므로, 정책에 따른 일관성을 유지할 수 있다. 특히 탐험(exploration) 중에도 안정적으로 학습이 가능하다.<br>
▣ 장점 : 에이전트의 실제 정책을 기반으로 학습하므로 정책의 일관성을 유지할 수 있으며, Q-learning보다 안정적인 성능을 낼 수 있다.<br>
▣ 단점 : Q-learning보다 수렴 속도가 느릴 수 있으며, 잘못된 정책을 사용할 경우 학습 성능이 떨어질 수 있다.<br>
▣ 응용분야 : 게임, 로봇 제어, 자율 시스템, 물류 최적화.<br>
▣ 모델식 : SARSA 업데이트 식, Q(s,a)는 상태 𝑠에서 행동 𝑎를 선택할 때의 Q값, 𝑎′ 는 다음 상태에서 선택된
▣ 주요 알고리즘 : 
▣ python 예제 : 

    import numpy as np

    # SARSA 알고리즘을 위한 환경 설정
    n_states = 5
    n_actions = 2
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1  # 학습률
    gamma = 0.9  # 할인 계수
    epsilon = 0.1  # 탐험 확률

    def choose_action(state):
    if np.random.uniform(0, 1) < epsilon: return np.random.choice(n_actions)
    else: return np.argmax(Q[state, :])

    def update_q(state, action, reward, next_state, next_action):
        predict = Q[state, action]
        target = reward + gamma * Q[next_state, next_action]
        Q[state, action] = predict + alpha (target - predict)

    # 예시 학습 반복
    for episode in range(100):
        state = np.random.randint(0, n_states)
        action = choose_action(state)
        while state != 4:  # 종료 상태 가정
            next_state = np.random.randint(0, n_states)
            reward = 1 if next_state == 4 else 0
            next_action = choose_action(next_state)
            update_q(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
    print(Q)

<br>

# [4] 유전 알고리즘(Genetic Algorithm)
▣ 정의 : 유전 알고리즘(Genetic Algorithm, GA)은 자연 선택과 유전학의 원리에 기반한 최적화 알고리즘으로, 개체군(population) 내에서 개체들이 적응도(fitness)에 따라 선택되고, 교차(crossover)와 돌연변이(mutation)를 통해 새로운 세대를 형성하여 최적 해를 찾아가는 과정이다.<br>
▣ 필요성 : 해 공간이 매우 크거나 복잡한 문제에서 전통적인 탐색 방법으로는 최적해를 찾기 어렵기 때문에, 진화 과정을 모방한 유전 알고리즘을 사용해 빠르고 효율적으로 최적화할 수 있다.<br>
▣ 장점 : 다양한 해를 동시에 탐색하므로 전역 최적해에 도달할 가능성이 높으며, 문제의 구조에 대한 구체적인 지식 없이도 적용할 수 있고, 비선형 문제나 다목적 최적화 문제에도 적합하다.<br>
▣ 단점 : 계산 비용이 많이 들 수 있으며, 특정 문제에서는 수렴이 느릴 수 있으며, 너무 빠르게 수렴하면 지역 최적해에 갇힐 가능성이 있다.<br>
▣ 응용분야 : 최적화 문제, 로봇 공학, 머신 러닝에서의 하이퍼파라미터 튜닝, 경로 계획, 게임 디자인, 재무 최적화 등.<br>
▣ 모델식 : 유전 알고리즘의 일반적인 구성 요소:
적응도(fitness): 해의 품질을 측정.
선택(selection): 높은 적응도를 가진 개체를 우선적으로 선택.
교차(crossover): 두 부모로부터 자손을 생성.
돌연변이(mutation): 자손의 일부 유전자를 무작위로 변형.
▣ 주요 알고리즘 : 초기화: 개체군을 무작위로 생성.
적응도 계산: 각 개체의 적응도를 계산.
선택: 적응도에 따라 부모 개체를 선택.
교차 및 돌연변이: 자손을 생성.
적응도 평가 후, 최적화될 때까지 반복.
▣ python 예제 : 

    import numpy as np

    def fitness(x):
        return np.sum(x)  # 최대화할 함수 (예시)

    def selection(pop, scores, k=3):
        selected_idx = np.random.choice(len(pop), k, replace=False)
        return pop[selected_idx[np.argmax(scores[selected_idx])]]

    def crossover(p1, p2, r_cross):
        if np.random.rand() < r_cross:
            pt = np.random.randint(1, len(p1))
            return np.hstack((p1[:pt], p2[pt:]))
        return p1

    def mutation(bitstring, r_mut):
        for i in range(len(bitstring)):
            if np.random.rand() < r_mut:
                bitstring[i] = 1 - bitstring[i]

    # 초기화
    n_bits = 10
    n_pop = 20
    r_cross = 0.9
    r_mut = 1.0 / n_bits
    n_iter = 100
    pop = np.random.randint(0, 2, (n_pop, n_bits))

    for gen in range(n_iter):
        scores = np.array([fitness(c) for c in pop])
        best = pop[np.argmax(scores)]
        print(f"Generation {gen}, Best: {best}, Fitness: {np.max(scores)}")
    
        new_pop = []
        for _ in range(n_pop // 2):
            p1, p2 = selection(pop, scores), selection(pop, scores)
            child1, child2 = crossover(p1, p2, r_cross), crossover(p2, p1, r_cross)
            mutation(child1, r_mut)
            mutation(child2, r_mut)
            new_pop += [child1, child2]
        pop = np.array(new_pop)
        
<br>

# [5] Asynchronous Advantage Actor-Critic(A3C)

<br>

---

# 앙상블 학습(Ensemble Learning, EL)
▣ API : https://scikit-learn.org/stable/api/sklearn.ensemble.html<br>
▣ 정의 : 앙상블 학습이란 다수의 기초 알고리즘(base algorithm)을 결합하여 더 나은 성능의 예측 모델을 형성하는 것을 말하며, 사용 목적에 따라 배깅(Bagging), 부스팅(Boosting), 스택킹(Stacking)으로 분류된다.<br>

# [6] 스태킹(Stacking)
▣ 정의 : 스태킹은 서로 다른 종류의 기반 모델(base model) 여러 개를 학습한 후, 이들의 예측 결과를 결합하는 방식이다. 개별 모델의 예측 결과를 다시 하나의 메타 모델(meta-model)로 학습시켜 최종 예측을 수행한다.<br>
▣ 필요성 : 단일 모델의 약점을 보완하기 위해 서로 다른 유형의 모델을 조합함으로써 더 나은 성능을 도출할 수 있다. 예를 들어, 결정 트리, 서포트벡터머신(SVM), 신경망 등 다양한 모델을 결합할 수 있다.<br>
▣ 장점 : 서로 다른 모델의 장점을 결합하여 더욱 강력한 예측 성능을 낼 수 있으며, 다양한 모델의 편향과 분산을 보완할 수 있다.<br>
▣ 단점 : 모델 조합이 복잡해질수록 계산 비용이 커지고, 메타 모델을 학습하는 데 추가적인 시간이 소요되며 과적합(overfitting)의 위험이 있다.<br>
▣ 응용분야 : 여러 모델의 특성이 유용할 때 사용한다. 예를 들어, 금융 예측, 이미지 분류 등 다양한 문제에서 활용된다.<br>
▣ 모델식 : $𝑓_1$ 은 각각의 개별 모델, $𝑓_2$ 는 메타 모델, $\widehat{y}=f_2(f_1(x_1),f_1(x_2),...f_1(x_n))$<br>
▣ python 예제 : 

    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

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

# [7] 배깅(Bagging)
▣ 정의 : 배깅은 동일한 모델을 여러 번 학습하되, 각 학습마다 다른 데이터 샘플을 사용한다. 주로 부트스트랩(bootstrap) 방법으로 샘플링된 데이터로 모델을 학습하며, 최종 예측은 개별 모델의 예측 결과를 평균 또는 투표로 결합한다. 대표적인 알고리즘은 랜덤 포레스트(Random Forest)이다.<br>
▣ 필요성 : 단일 모델이 데이터의 특정 부분에 과적합하는 것을 방지하고, 예측의 안정성을 높이기 위해 사용된다.<br>
▣ 장점 : 분산을 줄여 예측 성능을 향상시키며, 과적합(overfitting)을 방지하는 데 도움이 된다.<br>
▣ 단점 : 편향을 줄이는 데는 효과적이지 않을 수 있으며, 많은 모델을 학습하므로 계산 자원이 많이 필요하다.<br>
▣ 응용분야 : 랜덤 포레스트처럼 결정 트리 기반 모델에서 이미지 분류, 텍스트 분류, 금융 예측 등에 널리 사용된다.<br>
▣ 모델식 :  $𝑓_1$ 은 각각의 개별 모델, $\widehat{y}=\frac{1}{N}\sum_{i=1}^{N}f_i(x)$<br>
▣ python 예제 : 

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split


    # 데이터 로드 및 분할
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 배깅 모델 정의
    bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    bagging_clf.fit(X_train, y_train)

    # 성능 평가
    print(bagging_clf.score(X_test, y_test))

<br>

# [8] 부스팅(Boosting)
▣ 정의 : 부스팅은 약한 학습기(weak learner)를 연속적으로 학습시키며, 이전 학습에서 잘못 예측한 데이터에 가중치를 부여하여 다음 모델이 이를 더 잘 학습할 수 있도록 한다. 대표적인 알고리즘으로는 AdaBoost, Gradient Boosting, XGBoost 등이 있다.<br>
▣ 필요성 : 약한 학습기를 여러 번 반복하여 강력한 학습기를 만들 수 있으며, 특히 잘못된 예측에 집중하여 성능을 점진적으로 개선한다.<br>
▣ 장점 : 모델이 연속적으로 개선되기 때문에 높은 예측 성능을 보일 수 있으며, 오류를 줄이는 데 매우 효과적이다.<br>
▣ 단점 : 연속적인 학습 과정에서 모델이 과적합할 위험이 있으며, 학습 속도가 느릴 수 있다.<br>
▣ 응용분야 : 금융 예측, 분류 문제, 회귀 분석 등에서 많이 사용되며, 특히 XGBoost는 대회에서 많이 사용된다.<br>
▣ 모델식 : $f_i$ 는 약한 학습기, $𝛼_𝑖$ 는 각 학습기의 가중치, $\widehat{y}=\sum_{i=1}^{N}\alpha_i f_i(x)$<br>
▣ python 예제 : 

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # 데이터 로드 및 분할
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 부스팅 모델 정의
    boosting_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    boosting_clf.fit(X_train, y_train)

    # 성능 평가
    print(boosting_clf.score(X_test, y_test))

<br>

