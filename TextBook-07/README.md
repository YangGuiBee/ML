#  07 : CASE Study

---

  https://www.dbpia.co.kr/
  
  https://www.riss.kr/

  https://news.hada.io/search?q=ML
	
---  

# Do we need hundreds of classifiers to solve real world classification problems?

▣ 연구 목적 : 이 연구는 다양한 머신 러닝 분류 알고리즘들의 성능을 비교하여 특정 데이터셋에 대해 가장 성능이 좋은 분류기를 찾는 것을 목표로 하였다.<br> 

▣ 연구 방법론 : 179개의 분류기를 17개의 패밀리로 나누어 R, Weka, C, Matlab 등 다양한 소프트웨어 도구를 통해 분석하였다. 연구에 사용된 데이터셋은 UCI 머신러닝 저장소와 실제 문제 데이터 121개를 50%의 데이터로 파라미터 튜닝을 수행한 후, 교차 검증을 통해 성능을 평가하는 방식으로 진행하였다.<br>

▣ 연구 결과 : 랜덤 포레스트(Random Forest)가 대부분의 데이터셋에서 가장 좋은 성능을 기록했다. 특히, R에서 구현된 랜덤 포레스트 알고리즘은 최대 정확도의 94.1%를 달성했으며, 121개의 데이터셋 중 84.3%에서 90% 이상의 정확도를 보였다. SVM은 두 번째로 좋은 성능을 보였으며, Gaussian 커널을 사용한 SVM은 최대 정확도의 92.3%를 기록했다. 기타 모델 중에 신경망(MLP)과 부스팅(Boosting) 알고리즘도 높은 성능을 기록했다. 특히 C5.0, Extreme Learning Machine(ELM), 다층 퍼셉트론(MLP) 기반의 avNNet은 상위 성능 그룹에 포함되었다. 모든 알고리즘이 데이터셋에 따라 성능이 다르게 나타났으며, No-Free-Lunch 정리에 따라 어떤 알고리즘도 모든 데이터셋에서 최고의 성능을 보이지 않았다.<br>

▣ 결론 : 이 연구는 다수의 분류기를 비교함으로써 다양한 알고리즘 간의 성능 차이를 분석하고, 특정 알고리즘이 특정 상황에서 더 유리할 수 있음을 보여주었다. 연구 결과, 랜덤 포레스트와 SVM 계열의 알고리즘들이 전반적으로 우수한 성능을 보였으나, 각 데이터셋에 따라 최적의 알고리즘이 달라질 수 있음을 확인했다.<br>

▣ 분석대상 알고리즘 : 

| Family                           | 분류기 목록                                      |
|----------------------------------|---------------------------------------------------|
| 1. Discriminant Analysis (DA) | LDA, LDA2, RR-LDA, SDA, SLDA, Step-LDA, Step-QDA, Penalized-LDA, Sparse-LDA, QDA, QdaCov, Step-QDA, FDA, Step-FDA, PDA, RDA, MDA, HDDA |
| 2. Bayesian (BY) | Naive Bayes (NB), BayesNet, AODE, NaiveBayesSimple, NaiveBayesUpdateable, VBMPRadial |
| 3. Neural Networks (NNET) | MLP, avNNet, pcaNNet, RSNNS MLP, RSNNS RBF, RSNNS RBF-DDA, PNN, ELM, Cascade-Correlation (Cascor), Learning Vector Quantization (LVQ), Bidirectional Kohonen Map (BDK), DKP, DPP |
| 4. Support Vector Machines (SVM) | SVM (Gaussian), SVMlight, LibSVM, LibLINEAR, SMO, svmRadial, svmLinear, svmPoly, lssvmRadial |
| 5. Decision Trees (DT) | C5.0, J48, RandomTree, REPTree, DecisionStump, RandomSubspace, NBTree, RandomForest |
| 6. Rule-Based Classifiers (RL) | JRip (RIPPER), PART, OneR, ConjunctiveRule, Ridor, DTNB, ZeroR, DecisionTable |
| 7. Boosting (BST) | AdaBoost (with Decision Stump, J48), LogitBoost, MultiBoost (with DecisionStump, J48, RandomForest, Naive Bayes, OneR, PART), GradientBoosting |
| 8. Bagging (BAG) | Bagging (with Decision Stump, J48, Naive Bayes, OneR, RandomForest, Logistic, PART, Multilayer Perceptron), TreeBag, MetaCost, Bagging-SVM |
| 9. Stacking (STC) | Stacking, StackingC |
| 10. Random Forests (RF) | Random Forest (R, Weka), RRF (Regularized Random Forest), cForest, parRF, RotationForest |
| 11. Other Ensembles (OEN) | Random Committee, OrdinalClassClassifier, MultiScheme, MultiClassClassifier, Grading, Dagging, LWL (Locally Weighted Learning) |
| 12. Generalized Linear Models (GLM) | GLM (R), glmnet, bayesglm, glmStepAIC, mlm |
| 13. Nearest Neighbors (NN) | k-NN (R, Weka), IB1, IBk, NNge (Non-nested generalized exemplars) |
| 14. Partial Least Squares (PLSR) | PLSR, KernelPLS, WideKernelPLS, SparsePLS, Generalized PLS |
| 15. Logistic Regression (LR) | Logistic Regression (Logistic, multinom, SimpleLogistic) |
| 16. Multivariate Adaptive Regression Splines (MARS) | MARS (earth), gcvEarth |
| 17. Other Methods (OM) | PAM, VFI (Voting Feature Intervals), HyperPipes, FilteredClassifier, CVParameterSelection, ClassificationViaRegression, KStar |

<br>

## 1. 판별분석(Discriminant Analysis, DA) 
두 개 또는 그 이상의 그룹(또는 군집 또는 모집단)이 사전에 알려져 있을 때, 새로운 관측값을 특성에 기초하여 이미 알려진 모집단 가운데 하나로 분류하는 기법이다.<br><br>
### 1-1. lda R(Linear Discriminant Analysis)
특성: 기본 선형 판별 분석, MASS 패키지의 lda 함수 사용<br>
장점: 계산 효율이 좋고 간단함<br>
단점: 입력 데이터가 정규 분포를 따를 경우에만 좋은 성능<br>
모델식: 선형 분류<br>
응용분야: 의료, 생명과학, 금융 데이터 분류 등<br>

### 1-2. lda2 t
특성: LDA 변형, 컴포넌트 수를 조정 가능<br>
장점: 더 많은 컴포넌트 활용 가능<br>
단점: 컴포넌트 수 조정이 필요함<br>
모델식: 선형 판별<br>
응용분야: 다중 클래스 분류 문제<br>

### 1-3. rrlda R(Robust Regularized LDA)<br>
특성: 정규화된 LDA, rrlda 패키지 사용<br>
장점: 잡음이 많은 데이터에 강건<br>
단점: 적절한 파라미터 튜닝이 필요<br>
모델식: 정규화된 선형 판별<br>
응용분야: 고차원 데이터 분류<br>

### 1-4. sda t(Shrinkage Discriminant Analysis)
특성: 축소 LDA, sda 패키지 사용<br>
장점: 변수 선택이 포함된 선형 판별<br>
단점: 데이터가 적을 경우 성능 저하<br>
모델식: 축소 기반 판별<br>
응용분야: 유전자 데이터 분석<br>

### 1-5. slda t(Spherical LDA)
특성: 구형 기반 LDA, ipred 패키지 사용<br>
장점: 구형으로 분포된 데이터를 처리<br>
단점: 복잡한 데이터에 적용 어려움<br>
모델식: 구형 선형 판별<br>
응용분야: 간단한 데이터 분류<br>

### 1-6. stepLDA t
특성: 단계적 특징 선택 포함, klaR 패키지 사용<br>
장점: 단계별로 변수 선택 가능<br>
단점: 계산 시간이 많이 소요됨<br>
모델식: 특징 선택 기반 LDA<br>
응용분야: 피처가 많은 데이터셋<br>

### 1-7. sddaLDA R(Stepwise Diagonal Discriminant Analysis)
특성: 대각 판별 분석, SDDA 패키지 사용<br>
장점: 입력 변수 추가 가능<br>
단점: 변수 상관관계를 무시<br>
모델식: 대각 기반 판별<br>
응용분야: 다변량 데이터 분류<br>

### 1-8. PenalizedLDA t
특성: Lasso 제약 포함, penalizedLDA 패키지 사용<br>
장점: 고차원 데이터 분류<br>
단점: 과적합 위험<br>
모델식: Lasso 제약 기반<br>
응용분야: 이미지 분류, 유전자 데이터<br>

### 1-9. sparseLDA R
특성: 드문 판별 분석, sparseLDA 패키지 사용<br>
장점: 드문 데이터 처리에 강점<br>
단점: 파라미터 튜닝 어려움<br>
모델식: 드문 판별<br>
응용분야: 텍스트 데이터 분류<br>

### 1-10. qda t(Quadratic Discriminant Analysis)
특성: 2차 판별 분석, qda 함수 사용<br>
장점: 선형적으로 분리되지 않은 데이터에 효과적<br>
단점: 다중 공선성에 취약<br>
모델식: 2차 판별<br>
응용분야: 복잡한 분류 문제<br>

### 1-11. QdaCov t
특성: 강건한 QDA, rrcov 패키지 사용<br>
장점: 이상치에 강함<br>
단점: 계산량이 많음<br>
모델식: 강건한 2차 판별<br>
응용분야: 금융 데이터 분석<br>

### 1-12. sddaQDA R
특성: 단계적 2차 판별 분석, SDDA 패키지 사용<br>
장점: 단계적으로 변수 선택 가능<br>
단점: 계산 복잡성<br>
모델식: 2차 판별<br>
응용분야: 다중 클래스 분류<br>

### 1-13. stepQDA t
특성: 변수 선택 포함 QDA, klaR 패키지 사용<br>
장점: QDA의 유연성<br>
단점: 파라미터 튜닝 복잡<br>
모델식: 단계적 2차 판별<br>
응용분야: 복잡한 데이터셋<br>

### 1-14. fda R(Flexible Discriminant Analysis)
특성: 유연한 판별 분석, fda 함수 사용<br>
장점: 다양한 데이터셋에 적용 가능<br>
단점: 계산 복잡성<br>
모델식: 유연한 판별<br>
응용분야: 의학적 데이터 분석<br>

### 1-15. fda t
특성: 선형 회귀 기반 fda, caret 사용<br>
장점: 파라미터 튜닝 가능<br>
단점: 복잡한 데이터에 적합하지 않음<br>
모델식: 선형 회귀<br>
응용분야: 간단한 회귀 문제<br>

### 1-16. mda R(Mixture Discriminant Analysis)
특성: 혼합 모델 기반 판별 분석, mda 패키지 사용<br>
장점: 혼합 모델 처리<br>
단점: 과적합 가능성<br>
모델식: 혼합 모델<br>
응용분야: 유전자 데이터 분석<br>

### 1-17. mda t
특성: 튜닝 가능 mda, caret 사용<br>
장점: 파라미터 튜닝<br>
단점: 복잡한 구조<br>
모델식: 혼합 모델<br>
응용분야: 다중 클래스 문제<br>

### 1-18. pda t(Penalized Discriminant Analysis)
특성: 제약이 포함된 판별 분석, mda 패키지 사용<br>
장점: 과적합 방지<br>
단점: 과적합 가능성<br>
모델식: 제약 기반<br>
응용분야: 고차원 문제<br>

### 1-19. rda R (Regularized Discriminant Analysis)
특성: 정규화된 판별 분석, klaR 패키지 사용<br>
장점: 다중 공선성 해결<br>
단점: 복잡한 파라미터 조정<br>
모델식: 정규화된 판별<br>
응용분야: 복잡한 데이터<br>

### 1-20. hdda R (High-dimensional Discriminant Analysis)
특성: 고차원 데이터를 처리, HDclassif 사용<br>
장점: 고차원 공간에서의 분류<br>
단점: 계산 복잡성<br>
모델식: 고차원 판별<br>
응용분야: 이미지, 유전자 데이터 분석<br>
<br>

## 2. Bayesian(BY)
### 2-1. NaiveBayes R
특성: R의 klaR 패키지에서 제공하는 Naive Bayes 분류기<br>
장점: 간단하고 빠르며, 작은 데이터에서도 성능이 안정적임<br>
단점: 속성 간 독립 가정이 위배될 경우 성능 저하<br>
모델식: 가우시안 커널, 대역폭 1, 라플라스 보정 2 사용<br>
응용분야: 문서 분류, 이메일 스팸 필터링 등<br>

### 2-2. vbmpRadial t
특성: 변분 베이지안 다중항 프로빗 회귀, 가우시안 프로세스 사전 사용<br>
장점: 커널 기반 회귀로 복잡한 데이터 처리 가능<br>
단점: 계산 복잡성 높음<br>
모델식: vbmp 패키지를 사용한 반경 기저 함수 커널<br>
응용분야: 고차원 데이터에서의 분류 문제<br>

### 2-3. NaiveBayes w
특성: 훈련 데이터 분석을 통해 추정 정밀도 값을 선택하여 사용<br>
장점: 계산 효율이 좋음<br>
단점: 독립 가정 위배 시 성능 저하<br>
모델식: 추정 값을 선택하여 모델 구성<br>
응용분야: 분류 문제, 스팸 필터링<br>

### 2-4. NaiveBayesUpdateable w
특성: 반복적으로 훈련 패턴을 사용해 추정 정밀도 값을 업데이트<br>
장점: 학습 중 지속적으로 개선<br>
단점: 계산 복잡도가 높음<br>
모델식: 초기 값부터 시작하여 훈련 패턴으로 정밀도 값을 갱신<br>
응용분야: 실시간 분류 작업<br>

### 2-5. BayesNet w
특성: 베이지안 분류기의 앙상블, K2 탐색 방법 사용<br>
장점: 부모 노드와의 상관관계를 기반으로 확률표 학습<br>
단점: 복잡한 탐색 과정<br>
모델식: K2 탐색 방법, simpleEstimator를 사용한 베이지안 네트워크<br>
응용분야: 유전자 분석, 구조적 학습<br>

### 2-6. NaiveBayesSimple w
특성: 간단한 Naive Bayes 분류기, 숫자형 속성을 위한 정규 분포 모델 사용<br>
장점: 매우 간단하고 빠름<br>
단점: 속성 간 독립 가정 위배 시 성능 저하<br>
모델식: 정규 분포 사용<br>
응용분야: 간단한 분류 문제<br>
<br>

## 3. Neural Networks(NNET)
### 3-1. rbf m(Radial Basis Functions - Matlab)
특성: Matlab의 RBF 신경망, 가우시안 기저 함수 사용<br>
장점: 신경망 크기를 동적으로 조정 가능<br>
단점: 적절한 확산 값을 설정해야 함<br>
모델식: 가우시안 기저 함수 기반<br>
응용분야: 패턴 인식, 신호 처리<br>

### 3-2. rbf t(Radial Basis Functions - caret)
특성: RSNNS 패키지의 RBF 신경망<br>
장점: 쉬운 튜닝 가능<br>
단점: 모델 복잡성<br>
모델식: RBF 기반<br>
응용분야: 고차원 데이터 분류<br>

### 3-3. RBFNetwork w
특성: K-means로 중심을 선택하여 선형 회귀를 통해 학습<br>
장점: 비선형 분류에 적합<br>
단점: 계산 복잡<br>
모델식: 선형 회귀 기반<br>
응용분야: 패턴 인식, 이미지 처리<br>

### 3-4. rbfDDA t
특성: 동적 감쇠 조정을 사용하는 RBF 신경망<br>
장점: 네트워크 크기 자동 조정<br>
단점: 복잡한 네트워크 성장<br>
모델식: 동적 감쇠 조정<br>
응용분야: 신호 처리, 패턴 인식<br>

### 3-5. mlp m(Multi-Layer Perceptron - Matlab)
특성: Matlab의 다층 퍼셉트론<br>
장점: 다양한 문제에 적용 가능<br>
단점: 네트워크 크기 조정 필요<br>
모델식: MLP 기반<br>
응용분야: 이미지 분류, 패턴 인식<br>

### 3-6. mlp C
특성: FANN 라이브러리 기반 MLP<br>
장점: 다양한 훈련 알고리즘 지원<br>
단점: 훈련 알고리즘에 따른 성능 차이<br>
모델식: 다층 퍼셉트론 기반<br>
응용분야: 다양한 머신러닝 문제<br>

### 3-7. mlp t
특성: RSNNS 패키지를 사용한 MLP<br>
장점: 튜닝 용이<br>
단점: 파라미터 조정 복잡<br>
모델식: MLP 기반<br>
응용분야: 고차원 데이터 분류<br>

### 3-8. avNNet t
특성: MLP 앙상블로 구성<br>
장점: 낮은 연산 비용으로 성능 향상<br>
단점: 작은 뉴런 수에서 성능 저하<br>
모델식: MLP 앙상블 기반<br>
응용분야: 회귀 분석, 분류 문제<br>

### 3-9. mlpWeightDecay t
특성: 가중치 감소 MLP<br>
장점: 과적합 방지<br>
단점: 튜닝 필요<br>
모델식: 가중치 감소 기반<br>
응용분야: 고차원 데이터 처리<br>

### 3-10. nnet t
특성: caret 패키지를 사용한 nnet 함수<br>
장점: 간편한 사용<br>
단점: 네트워크 크기 및 가중치 조정 필요<br>
모델식: MLP 기반<br>
응용분야: 회귀 분석<br>

### 3-11. pcaNNet t
특성: PCA를 적용한 MLP<br>
장점: 차원 축소로 성능 향상<br>
단점: 데이터 전처리 필요<br>
모델식: PCA 기반<br>
응용분야: 고차원 데이터 분류<br>

### 3-12. MultilayerPerceptron w
특성: Weka에서 제공하는 MLP<br>
장점: 다양한 문제에 적용 가능<br>
단점: 학습 속도가 느림<br>
모델식: MLP 기반<br>
응용분야: 이미지 처리<br>

### 3-13. pnn m(Probabilistic Neural Network)
특성: 확률적 신경망<br>
장점: 간단하고 빠름<br>
단점: 스프레드 값 튜닝 필요<br>
모델식: 가우시안 기저 함수 기반<br>
응용분야: 분류 문제<br>

### 3-14. elm m(Extreme Learning Machine)
특성: 빠른 학습 속도를 가진 ELM<br>
장점: 매우 빠른 학습<br>
단점: 파라미터 조정 필요<br>
모델식: ELM 기반<br>
응용분야: 실시간 데이터 처리<br>

### 3-15. elm kernel m
특성: 가우시안 커널을 사용하는 ELM<br>
장점: 비선형 문제 해결<br>
단점: 파라미터 튜닝 필요<br>
모델식: 커널 기반 ELM<br>
응용분야: 고차원 데이터 분류<br>

### 3-16. cascor C(Cascade Correlation Network)
특성: 계단형 상관 신경망<br>
장점: 높은 유연성<br>
단점: 복잡한 구조<br>
모델식: 계단형 상관 기반<br>
응용분야: 패턴 인식<br>

### 3-17. lvq R(Learning Vector Quantization)
특성: 벡터 양자화를 사용한 학습<br>
장점: 고차원 데이터에 적합<br>
단점: 파라미터 조정 필요<br>
모델식: 벡터 양자화 기반<br>
응용분야: 패턴 인식<br>

### 3-18. lvq t
특성: LVQ 알고리즘을 사용<br>
장점: 튜닝 가능<br>
단점: 복잡한 튜닝 과정<br>
모델식: LVQ 기반<br>
응용분야: 분류 문제<br>

### 3-19. bdk R(Bi-directional Kohonen Map)
특성: 양방향 코호넨 맵<br>
장점: 시각화에 적합<br>
단점: 복잡한 데이터 처리에 어려움<br>
모델식: 양방향 맵 기반<br>
응용분야: 군집화 및 분류<br>

### 3-20. dkp C(Direct Kernel Perceptron)
특성: 커널 기반 퍼셉트론<br>
장점: 빠름<br>
단점: 일부 데이터에서 성능 저하<br>
모델식: 커널 퍼셉트론<br>
응용분야: 실시간 분류 문제<br>

### 3-21. dpp C(Direct Parallel Perceptron)
특성: 병렬 퍼셉트론<br>
장점: 빠른 연산<br>
단점: 고차원 데이터에서 성능 저하<br>
모델식: 병렬 퍼셉트론<br>
응용분야: 패턴 인식<br>

<br>

## 4. Support Vector Machines (SVM)
### 4-1. svm C (Support Vector Machine with Gaussian Kernel)
특성: LibSVM을 이용해 구현된 Gaussian 커널 기반 SVM. 정규화 파라미터 C와 커널 스프레드 감마 값 조정<br>
장점: 다중 클래스 데이터에서 one-vs-one 접근 방식 사용<br>
단점: 대규모 데이터에 계산 복잡도 높음<br>
모델식: Gaussian 커널 $K(x, x') = e^(-γ ||x - x'||^2)$<br>
응용분야: 이미지 인식, 생물정보학<br>

### 4-2. svmlight C
특성: C언어 기반으로 Joachims(1999)에 의해 개발된 SVM 구현체<br>
장점: 매우 널리 사용됨<br>
단점: 라이브러리가 아닌 커맨드라인에서만 사용 가능, 대규모 데이터에서 에러 발생 가능<br>
모델식: Gaussian 커널<br>
응용분야: 텍스트 분류<br>

### 4-3. LibSVM w
특성: Weka에서 호출되는 LibSVM 라이브러리 사용<br>
장점: 다양한 파라미터와 커널 지원<br>
단점: 설정 복잡성<br>
모델식: Gaussian 커널 기반 SVM<br>
응용분야: 다양한 머신러닝 문제<br>

### 4-4. LibLINEAR w
특성: LibLinear 라이브러리 사용, 대규모 고차원 데이터에 적합<br>
장점: 선형 모델에 효율적<br>
단점: 비선형 데이터 처리 어려움<br>
모델식: L2-loss 선형 SVM<br>
응용분야: 대규모 텍스트 데이터 처리<br>

### 4-5. svmRadial t
특성: kernlab 패키지의 Gaussian 커널 기반 SVM, C와 스프레드 조정 가능<br>
장점: 다양한 커널 사용 가능<br>
단점: 튜닝 필요<br>
모델식: Gaussian 커널 기반<br>
응용분야: 고차원 데이터 처리<br>

### 4-6. svmRadialCost t
특성: kernlab 패키지 사용, 커널 스프레드를 자동으로 계산<br>
장점: 자동 튜닝<br>
단점: 제한된 튜닝 옵션<br>
모델식: Gaussian 커널<br>
응용분야: 회귀 및 분류 문제<br>

### 4-7. svmLinear t
특성: kernlab 패키지에서 선형 커널 사용, C 값 조정 가능<br>
장점: 간단하고 빠름<br>
단점: 비선형 데이터에서 성능 저하<br>
모델식: 선형 커널 기반<br>
응용분야: 고차원 텍스트 데이터<br>

### 4-8. svmPoly t
특성: kernlab 패키지의 다항 커널 사용, 차수와 스케일, 오프셋 조정 가능<br>
장점: 다양한 비선형 관계 모델링<br>
단점: 계산 복잡성 높음<br>
모델식: 다항식 커널 $(sx^Ty + o)^d$<br>
응용분야: 복잡한 데이터 분류<br>

### 4-9. lssvmRadial t
특성: 최소 제곱 SVM, Gaussian 커널 기반<br>
장점: 간단한 구현<br>
단점: 성능 낮음<br>
모델식: Gaussian 커널 기반 최소 제곱 SVM<br>
응용분야: 회귀 및 분류 문제<br>

### 4-10. SMO w
특성: 순차 최소화 최적화(SMO)를 이용한 SVM, one-vs-one 다중 클래스 처리 방식 사용<br>
장점: 다중 클래스 문제에서 효율적<br>
단점: 복잡한 파라미터 튜닝 필요<br>
모델식: 이차 커널 기반 SVM<br>
응용분야: 텍스트 및 이미지 분류<br>
<br>

## 5. Decision Trees (DT)
### 5-1. rpart R
특성: rpart 패키지를 이용한 재귀적 분할, 변수 분할 시 엔트로피 또는 지니 지수를 사용<br>
장점: 쉽게 해석 가능, 다양한 파라미터 튜닝 가능<br>
단점: 깊은 트리는 과적합 위험<br>
모델식: 결정 트리 기반<br>
응용분야: 의료 데이터 분석, 고객 분류<br>

### 5-2. rpart t
특성: 복잡도 매개변수 조정으로 트리의 정확도 증가, rpart 함수 사용<br>
장점: 튜닝 가능성<br>
단점: 매개변수 설정에 따라 성능 변동<br>
모델식: 트리 복잡도 조정<br>
응용분야: 분류 및 회귀 문제<br>

### 5-3. rpart2 t
특성: rpart 함수를 이용해 트리 깊이를 최대 10까지 조정<br>
장점: 깊이 조정 가능<br>
단점: 튜닝이 복잡할 수 있음<br>
모델식: 트리 깊이 제한<br>
응용분야: 데이터 마이닝<br>

### 5-4. obliqueTree R
특성: oblique.tree 패키지로 구현, 이진 재귀적 분할, 선형 결합<br>
장점: 선형 결합을 통한 분할 가능<br>
단점: 계산 복잡성 증가<br>
모델식: 이진 분할<br>
응용분야: 고차원 데이터 분류<br>

### 5-5. C5.0Tree t
특성: C5.0 패키지를 사용해 트리 구성, 파라미터 조정 없이 단일 트리 생성<br>
장점: 빠르고 효율적<br>
단점: 복잡한 데이터에서 성능 저하 가능<br>
모델식: C5.0 결정 트리<br>
응용분야: 텍스트 및 데이터 마이닝<br>

### 5-6. ctree t
특성: 조건부 추론 트리, 통계적 테스트 기반 분할<br>
장점: 변수의 통계적 유의성을 반영<br>
단점: 복잡한 설정 필요<br>
모델식: 조건부 추론 트리<br>
응용분야: 생명과학 데이터 분석<br>

### 5-7. ctree2 t
특성: ctree 함수 사용, 최대 깊이를 조정해 트리 생성<br>
장점: 트리 깊이 조정 가능<br>
단점: 트리 깊이에 따른 성능 변동<br>
모델식: 조건부 추론 트리<br>
응용분야: 다차원 데이터 분석<br>

### 5-8. J48 w
특성: C4.5 가지치기 트리, Weka에서 제공<br>
장점: 가지치기 기능으로 과적합 방지<br>
단점: 작은 데이터셋에서 성능 저하<br>
모델식: C4.5 기반<br>
응용분야: 텍스트 분류<br>

### 5-9. J48 t
특성: Weka의 J48 함수 사용, C5.0 가지치기 트리<br>
장점: 쉽게 해석 가능<br>
단점: 복잡한 문제에서 과적합 위험<br>
모델식: C5.0 트리 기반<br>
응용분야: 데이터 마이닝<br>

### 5-10. RandomSubSpace w
특성: 랜덤 서브스페이스를 사용한 REPTree 학습<br>
장점: 다수의 속성을 사용해 다양성 증가<br>
단점: 계산 복잡성 높음<br>
모델식: 랜덤 속성 기반<br>
응용분야: 이미지 처리<br>

### 5-11. NBTree w
특성: 나이브 베이즈 분류기를 잎 노드로 사용하는 결정 트리<br>
장점: 트리와 베이즈 결합으로 성능 향상<br>
단점: 설정 복잡성<br>
모델식: 혼합 모델 기반<br>
응용분야: 패턴 인식<br>

### 5-12. RandomTree w
특성: 무한 트리 깊이로 학습, 무작위 입력 선택<br>
장점: 매우 유연한 모델<br>
단점: 과적합 위험<br>
모델식: 무작위 트리 기반<br>
응용분야: 데이터 마이닝<br>

### 5-13. REPTree w
특성: 정보 이득을 사용한 가지치기 트리 학습<br>
장점: 효율적이며 가지치기 기능<br>
단점: 큰 데이터셋에서 성능 저하<br>
모델식: 정보 이득 기반<br>
응용분야: 텍스트 분류<br>

### 5-14. DecisionStump w
특성: 한 개의 노드로 분류하는 결정 트리<br>
장점: 매우 간단하며 빠름<br>
단점: 복잡한 데이터에 적합하지 않음<br>
모델식: 한 노드 트리<br>
응용분야: 단순 회귀 및 분류<br>
<br>

## 6. Rule-Based Classifiers (RL)

<br>

## 7. Boosting (BST)

<br>

## 8. Bagging (BAG)

<br>

## 9. Stacking (STC)

<br>

## 10. Random Forests (RF)

<br>

## 11. Other Ensembles (OEN)

<br>

## 12. Generalized Linear Models (GLM)

<br>

## 13. Nearest Neighbors (NN)

<br>

## 14. Partial Least Squares (PLSR)

<br>

## 15. Logistic Regression (LR)

<br>

## 16. Multivariate Adaptive Regression Splines (MARS)

<br>

## 17. Other Methods (OM)

<br>


