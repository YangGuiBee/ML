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
| 1. Discriminant Analysis(DA) : 20 | LDA R, LDA2 t, RRLDA R, SDA t, SLDA t, stepLDA t, SDDA R, PenalizedLDA t, sparseLDA R, QDA t, QDACov t, SDDAQDA R, stepQDA t, FDA R, FDA t, MDA R, MDA t, PDA t, RDA R, HDDA R |
| 2. Bayesian(BY) : 6 | NaiveBayes R, vbmpRadial t, NaiveBayes w, NaiveBayesUpdateable w, BayesNet w, NaiveBayesSimple w
 |
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
| 15. Logistic and multinomial regression (LMR) | Logistic Regression (Logistic, multinom, SimpleLogistic) |
| 16. Multivariate Adaptive Regression Splines (MARS) | MARS (earth), gcvEarth |
| 17. Other Methods (OM) | PAM, VFI (Voting Feature Intervals), HyperPipes, FilteredClassifier, CVParameterSelection, ClassificationViaRegression, KStar |

<br>

## 1. 판별분석(Discriminant Analysis, DA) 
두 개 또는 그 이상의 그룹(또는 군집 또는 모집단)이 사전에 알려져 있을 때, 새로운 관측값을 특성에 기초하여 이미 알려진 모집단 가운데 하나로 분류하는 기법이다.<br><br>
### 1-1. LDA(Linear Discriminant Analysis) R
특성: 기본 선형 판별 분석, MASS 패키지의 lda 함수 사용<br>
장점: 계산 효율이 좋고 간단함<br>
단점: 입력 데이터가 정규 분포를 따를 경우에만 좋은 성능<br>
모델식: 선형 분류<br>
응용분야: 의료, 생명과학, 금융 데이터 분류 등<br>

### 1-2. LDA2 t
특성: LDA 변형, 컴포넌트 수를 조정 가능<br>
장점: 더 많은 컴포넌트 활용 가능<br>
단점: 컴포넌트 수 조정이 필요함<br>
모델식: 선형 판별<br>
응용분야: 다중 클래스 분류 문제<br>

### 1-3. RRLDA(Robust Regularized LDA) R<br>
특성: 정규화된 LDA, rrlda 패키지 사용<br>
장점: 잡음이 많은 데이터에 강건<br>
단점: 적절한 파라미터 튜닝이 필요<br>
모델식: 정규화된 선형 판별<br>
응용분야: 고차원 데이터 분류<br>

### 1-4. SDA(Shrinkage Discriminant Analysis) t
특성: 축소 LDA, sda 패키지 사용<br>
장점: 변수 선택이 포함된 선형 판별<br>
단점: 데이터가 적을 경우 성능 저하<br>
모델식: 축소 기반 판별<br>
응용분야: 유전자 데이터 분석<br>

### 1-5. SLDA(Spherical LDA) t
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

### 1-7. SDDA(Stepwise Diagonal Discriminant Analysis) R
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

### 1-10. QDA(Quadratic Discriminant Analysis) t
특성: 2차 판별 분석, qda 함수 사용<br>
장점: 선형적으로 분리되지 않은 데이터에 효과적<br>
단점: 다중 공선성에 취약<br>
모델식: 2차 판별<br>
응용분야: 복잡한 분류 문제<br>

### 1-11. QDACov t
특성: 강건한 QDA, rrcov 패키지 사용<br>
장점: 이상치에 강함<br>
단점: 계산량이 많음<br>
모델식: 강건한 2차 판별<br>
응용분야: 금융 데이터 분석<br>

### 1-12. SDDAQDA R
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

### 1-14. FDA(Flexible Discriminant Analysis) R
특성: 유연한 판별 분석, fda 함수 사용<br>
장점: 다양한 데이터셋에 적용 가능<br>
단점: 계산 복잡성<br>
모델식: 유연한 판별<br>
응용분야: 의학적 데이터 분석<br>

### 1-15. FDA t
특성: 선형 회귀 기반 fda, caret 사용<br>
장점: 파라미터 튜닝 가능<br>
단점: 복잡한 데이터에 적합하지 않음<br>
모델식: 선형 회귀<br>
응용분야: 간단한 회귀 문제<br>

### 1-16. MDA(Mixture Discriminant Analysis) R
특성: 혼합 모델 기반 판별 분석, mda 패키지 사용<br>
장점: 혼합 모델 처리<br>
단점: 과적합 가능성<br>
모델식: 혼합 모델<br>
응용분야: 유전자 데이터 분석<br>

### 1-17. MDA t
특성: 튜닝 가능 mda, caret 사용<br>
장점: 파라미터 튜닝<br>
단점: 복잡한 구조<br>
모델식: 혼합 모델<br>
응용분야: 다중 클래스 문제<br>

### 1-18. PDA(Penalized Discriminant Analysis) t
특성: 제약이 포함된 판별 분석, mda 패키지 사용<br>
장점: 과적합 방지<br>
단점: 과적합 가능성<br>
모델식: 제약 기반<br>
응용분야: 고차원 문제<br>

### 1-19. RDA(Regularized Discriminant Analysis) R
특성: 정규화된 판별 분석, klaR 패키지 사용<br>
장점: 다중 공선성 해결<br>
단점: 복잡한 파라미터 조정<br>
모델식: 정규화된 판별<br>
응용분야: 복잡한 데이터<br>

### 1-20. HDDA(High-dimensional Discriminant Analysis) R
특성: 고차원 데이터를 처리, HDclassif 사용<br>
장점: 고차원 공간에서의 분류<br>
단점: 계산 복잡성<br>
모델식: 고차원 판별<br>
응용분야: 이미지, 유전자 데이터 분석<br>
<br>

## 2. Bayesian(BY)
베이즈 추론(Bayesian inference)은 통계적 추론의 한 방법으로, 추론 대상의 사전 확률과 추가적인 정보를 통해 해당 대상의 사후 확률을 추론하는 방법으로 베이즈 추론은 베이즈 확률론을 기반으로 하며, 이는 추론하는 대상을 확률변수로 보아 그 변수의 확률분포를 추정하는 것을 의미한다.<br>

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

## 6. Rule-Based Classifiers(RL)
### 6-1. PART w
특성: C4.5 결정 트리의 가지를 규칙으로 변환하여 부분적인 트리를 학습<br>
장점: 학습 데이터에 대해 가지치기를 사용해 규칙을 만들기 때문에 간결함<br>
단점: 일부 데이터에서는 성능이 낮을 수 있음<br>
모델식: C4.5 규칙 기반<br>
응용분야: 텍스트 분류, 데이터 마이닝<br>

### 6-2. PART t
특성: RWeka 패키지를 사용해 PART 학습, 가지치기 및 복잡도 매개변수 튜닝 가능<br>
장점: 가지치기 기능으로 과적합 방지<br>
단점: 매개변수 튜닝 필요<br>
모델식: PART 규칙 기반<br>
응용분야: 다양한 분류 작업<br>

### 6-3. C5.0Rules t
특성: C5.0 결정 트리를 학습하여 규칙 세트를 생성<br>
장점: 효율적이며 강력한 성능<br>
단점: 매우 복잡한 데이터에서는 성능 저하 가능<br>
모델식: C5.0 규칙 기반<br>
응용분야: 데이터 마이닝, 텍스트 분류<br>

### 6-4. JRip t
특성: RIPPER 알고리즘을 기반으로 규칙 세트를 학습<br>
장점: 반복적 가지치기를 통해 에러 감소<br>
단점: 일부 데이터셋에서 과적합 발생 가능<br>
모델식: 반복적 가지치기 규칙<br>
응용분야: 텍스트 분류, 의료 데이터 분석<br>

### 6-5. JRip w
특성: RIPPER 알고리즘을 사용해 두 번의 최적화 실행으로 학습<br>
장점: 빠르고 효율적<br>
단점: 매개변수 조정 필요<br>
모델식: RIPPER 기반<br>
응용분야: 데이터 마이닝<br>

### 6-6. OneR t
특성: OneR 알고리즘, 최소 오류를 기준으로 단일 규칙 사용<br>
장점: 매우 간단하고 빠름<br>
단점: 규칙이 하나만 있어 복잡한 문제에는 적합하지 않음<br>
모델식: OneR 기반<br>
응용분야: 간단한 분류 문제<br>

### 6-7. OneR w
특성: Weka에서 최소 6개의 객체를 버킷에 넣어 OneR 규칙을 학습<br>
장점: 빠르고 간단함<br>
단점: 작은 데이터셋에 적합하지 않음<br>
모델식: OneR 규칙<br>
응용분야: 텍스트 및 간단한 데이터 분류<br>

### 6-8. DTNB w
특성: 의사결정 테이블과 나이브 베이즈를 결합한 하이브리드 분류기<br>
장점: 두 알고리즘의 장점을 결합<br>
단점: 계산 복잡성<br>
모델식: 의사결정 테이블과 나이브 베이즈 혼합<br>
응용분야: 복잡한 데이터 분류<br>

### 6-9. Ridor w
특성: Ripple-Down 규칙 학습기, 최소 2개의 인스턴스를 가중치로 사용<br>
장점: 실시간 학습 가능<br>
단점: 성능 변동이 클 수 있음<br>
모델식: Ripple-Down 기반<br>
응용분야: 온라인 학습<br>

### 6-10. ZeroR w
특성: 모든 테스트 패턴에 대해 가장 많이 나타난 클래스를 예측<br>
장점: 매우 간단함<br>
단점: 성능이 낮음<br>
모델식: ZeroR 기반<br>
응용분야: 기준 성능 설정<br>

### 6-11. DecisionTable w
특성: 다수결 원칙을 사용하는 단순한 의사결정 테이블<br>
장점: 해석이 매우 쉬움<br>
단점: 복잡한 데이터에 적합하지 않음<br>
모델식: 다수결 기반<br>
응용분야: 간단한 데이터 분류<br>

### 6-12. ConjunctiveRule w
특성: 여러 조건을 AND로 결합한 단일 규칙 사용<br>
장점: 간단하고 계산 효율적<br>
단점: 일부 데이터에서 성능 저하 가능<br>
모델식: AND 규칙 기반<br>
응용분야: 텍스트 및 간단한 데이터 분류<br>

<br>

## 7. Boosting (BST)
### 7-1. adaboost R
특성: adabag 패키지를 사용한 Adaboost.M1 방식, 분류 트리 사용<br>
장점: 성능이 우수하고 계산 효율적<br>
단점: 매우 복잡한 데이터에서 성능 저하 가능<br>
모델식: Adaboost.M1<br>
응용분야: 분류 문제, 패턴 인식<br>

### 7-2. logitboost R
특성: caTools 패키지를 사용한 DecisionStump 기반 LogitBoost 앙상블<br>
장점: 반복적 학습으로 정확도 향상<br>
단점: 반복 횟수에 따라 성능이 변동됨<br>
모델식: LogitBoost<br>
응용분야: 회귀 분석, 분류 문제<br>

### 7-3. LogitBoost w
특성: DecisionStump를 기반으로 한 가산적 로지스틱 회귀 사용<br>
장점: 빠른 계산 속도<br>
단점: 내부 교차 검증 필요<br>
모델식: 로지스틱 회귀<br>
응용분야: 분류 문제, 데이터 분석<br>

### 7-4. RacedIncrementalLogitBoost w
특성: LogitBoost와 증분 학습을 사용한 레이싱 기법, 검증 세트 포함<br>
장점: 대용량 데이터셋에서 성능 향상<br>
단점: 검증 세트가 필요함<br>
모델식: 증분 학습 기반<br>
응용분야: 실시간 데이터 처리<br>

### 7-5. AdaBoostM1 DecisionStump w
특성: DecisionStump 기반의 Adaboost.M1<br>
장점: 약한 분류기를 사용해 성능 향상<br>
단점: 매우 복잡한 문제에 한계 있음<br>
모델식: Adaboost.M1<br>
응용분야: 텍스트 분류<br>

### 7-6. AdaBoostM1 J48 w
특성: J48 분류기를 사용하는 Adaboost.M1<br>
장점: 복잡한 데이터에서도 성능 유지<br>
단점: J48 성능에 크게 의존<br>
모델식: Adaboost.M1<br>
응용분야: 이미지 처리, 텍스트 분류<br>

### 7-7. C5.0 t
특성: C5.0 의사결정 트리를 기반으로 Boosting 앙상블 학습<br>
장점: 성능이 우수하고 다양한 기능 제공<br>
단점: 파라미터 튜닝 복잡성<br>
모델식: C5.0 Boosting<br>
응용분야: 고차원 데이터 처리<br>

### 7-8. MultiBoostAB DecisionStump w
특성: Adaboost와 Wagging을 결합한 MultiBoost 앙상블<br>
장점: 다양한 기법을 결합해 성능 향상<br>
단점: 복잡한 데이터 처리 시 계산 시간이 길어짐<br>
모델식: MultiBoost<br>
응용분야: 데이터 마이닝<br>

### 7-9. MultiBoostAB DecisionTable w
특성: DecisionTable을 사용하는 MultiBoost 앙상블<br>
장점: 매우 간단하고 계산 효율적<br>
단점: 복잡한 데이터에서는 성능 저하 가능<br>
모델식: MultiBoost<br>
응용분야: 간단한 데이터 처리<br>

### 7-10. MultiBoostAB IBk w
특성: IBk 분류기를 사용하는 MultiBoost<br>
장점: 가까운 이웃 기반으로 높은 정확도 제공<br>
단점: 복잡한 데이터에서 성능 저하<br>
모델식: IBk 기반 MultiBoost<br>
응용분야: 패턴 인식, 이미지 처리<br>

### 7-11. MultiBoostAB J48 w
특성: J48 분류기를 사용하는 MultiBoost<br>
장점: 가지치기 기능을 통해 과적합 방지<br>
단점: 복잡한 데이터에서 성능 변동<br>
모델식: MultiBoost<br>
응용분야: 텍스트 및 데이터 마이닝<br>

### 7-12. MultiBoostAB LibSVM w
특성: LibSVM 분류기를 사용하는 MultiBoost<br>
장점: 강력한 분류 성능<br>
단점: 비선형 데이터에서 계산 복잡성<br>
모델식: LibSVM 기반 MultiBoost<br>
응용분야: 고차원 데이터 분석<br>

### 7-13. MultiBoostAB Logistic w
특성: 로지스틱 회귀를 사용하는 MultiBoost<br>
장점: 빠르고 정확한 분류<br>
단점: 특정 데이터에 성능 변동<br>
모델식: 로지스틱 회귀 기반 MultiBoost<br>
응용분야: 회귀 분석<br>

### 7-14. MultiBoostAB MultilayerPerceptron w
특성: 다층 퍼셉트론(MLP)을 사용하는 MultiBoost<br>
장점: 비선형 데이터 처리 능력 우수<br>
단점: 계산 비용이 큼<br>
모델식: MLP 기반 MultiBoost<br>
응용분야: 이미지 및 텍스트 분류<br>

### 7-15. MultiBoostAB NaiveBayes w
특성: 나이브 베이즈 분류기를 사용하는 MultiBoost<br>
장점: 계산 속도가 빠름<br>
단점: 독립성 가정이 위배될 경우 성능 저하<br>
모델식: NaiveBayes 기반 MultiBoost<br>
응용분야: 간단한 데이터 처리<br>

### 7-16. MultiBoostAB OneR w
특성: OneR 분류기를 사용하는 MultiBoost<br>
장점: 매우 간단하고 빠름<br>
단점: 복잡한 데이터에는 적합하지 않음<br>
모델식: OneR 기반 MultiBoost<br>
응용분야: 간단한 분류 문제<br>

### 7-17. MultiBoostAB PART w
특성: PART 분류기를 사용하는 MultiBoost<br>
장점: 규칙 기반 학습으로 성능 향상<br>
단점: 계산 복잡성 높음<br>
모델식: PART 기반 MultiBoost<br>
응용분야: 데이터 마이닝<br>

### 7-18. MultiBoostAB RandomForest w
특성: 랜덤 포레스트를 사용하는 MultiBoost<br>
장점: 높은 정확도 제공<br>
단점: 자체적으로 강력한 모델이기에 사용 효율이 떨어짐<br>
모델식: RandomForest 기반 MultiBoost<br>
응용분야: 고차원 데이터 분석<br>

### 7-19. MultiBoostAB RandomTree w
특성: 무작위 트리를 사용하는 MultiBoost<br>
장점: 다양한 트리 기반 모델 생성 가능<br>
단점: 성능이 일정하지 않음<br>
모델식: RandomTree 기반 MultiBoost<br>
응용분야: 분류 및 회귀 문제<br>

### 7-20. MultiBoostAB REPTree w
특성: REPTree 기반 MultiBoost<br>
장점: 가지치기 기능으로 과적합 방지<br>
단점: 복잡한 데이터에 약함<br>
모델식: REPTree 기반 MultiBoost<br>
응용분야: 텍스트 분류<br>

<br>

## 8. Bagging (BAG)
### 8-1. bagging R
특성: ipred 패키지를 사용한 결정 트리의 배깅(ensemble)<br>
장점: 다양한 데이터에서 성능 우수<br>
단점: 배깅의 계산 복잡성<br>
모델식: 배깅 기반<br>
응용분야: 다양한 분류 문제<br>

### 8-2. treebag t
특성: caret 인터페이스를 사용해 결정 트리의 배깅 학습<br>
장점: 튜닝 용이<br>
단점: 계산 비용이 클 수 있음<br>
모델식: 배깅 기반<br>
응용분야: 회귀 및 분류 문제<br>

### 8-3. ldaBag R
특성: LDA를 사용한 배깅 앙상블<br>
장점: 선형 분류에 적합<br>
단점: 비선형 데이터 처리 어려움<br>
모델식: LDA 기반 배깅<br>
응용분야: 고차원 데이터 분류<br>

### 8-4. plsBag R
특성: 부분 최소 제곱 회귀(PLS) 기반 배깅<br>
장점: 다변량 데이터에 적합<br>
단점: 계산 복잡성<br>
모델식: PLS 기반 배깅<br>
응용분야: 다중 회귀 분석<br>

### 8-5. nbBag R
특성: 나이브 베이즈 분류기 기반 배깅<br>
장점: 계산 효율적<br>
단점: 독립 가정이 위배될 때 성능 저하<br>
모델식: Naive Bayes 기반 배깅<br>
응용분야: 텍스트 분류<br>

### 8-6. ctreeBag R
특성: 조건부 추론 트리를 사용하는 배깅<br>
장점: 통계적 유의성을 고려한 분할<br>
단점: 복잡한 설정<br>
모델식: 조건부 추론 트리 기반 배깅<br>
응용분야: 생물학 데이터 분석<br>

### 8-7. svmBag R
특성: SVM 기반 배깅<br>
장점: 강력한 분류 성능<br>
단점: 계산 시간이 큼<br>
모델식: SVM 기반 배깅<br>
응용분야: 고차원 데이터 분석<br>

### 8-8. nnetBag R
특성: 다층 퍼셉트론(MLP) 기반 배깅<br>
장점: 비선형 데이터 처리에 적합<br>
단점: 훈련 시간이 길 수 있음<br>
모델식: MLP 기반 배깅<br>
응용분야: 패턴 인식, 이미지 처리<br>

### 8-9. MetaCost w
특성: 비용 민감 학습을 결합한 배깅<br>
장점: 오류 비용에 따라 모델 성능 최적화<br>
단점: 복잡한 설정 필요<br>
모델식: 비용 민감 배깅<br>
응용분야: 비용이 중요한 분류 문제<br>

### 8-10. Bagging DecisionStump w
특성: DecisionStump를 사용하는 배깅<br>
장점: 간단하고 빠름<br>
단점: 복잡한 데이터에 적합하지 않음<br>
모델식: DecisionStump 기반 배깅<br>
응용분야: 간단한 분류 문제<br>

### 8-11. Bagging DecisionTable w
특성: 결정 테이블 기반 배깅<br>
장점: 해석이 용이<br>
단점: 복잡한 데이터에서 성능 저하<br>
모델식: 결정 테이블 기반 배깅<br>
응용분야: 간단한 데이터 분류<br>

### 8-12. Bagging HyperPipes w
특성: HyperPipes를 사용하는 배깅<br>
장점: 비선형 데이터에 적합<br>
단점: 설정이 복잡할 수 있음<br>
모델식: HyperPipes 기반 배깅<br>
응용분야: 패턴 인식<br>

### 8-13. Bagging IBk w
특성: K-최근접 이웃(KNN) 기반 배깅<br>
장점: 다양한 거리 척도 사용 가능<br>
단점: 계산 복잡성<br>
모델식: KNN 기반 배깅<br>
응용분야: 패턴 인식, 이미지 분석<br>

### 8-14. Bagging J48 w
특성: J48 결정 트리 기반 배깅<br>
장점: 가지치기 기능을 통해 성능 최적화<br>
단점: 큰 데이터셋에서 성능 저하<br>
모델식: J48 기반 배깅<br>
응용분야: 데이터 마이닝<br>

### 8-15. Bagging LibSVM w
특성: Gaussian 커널을 사용하는 SVM 배깅<br>
장점: 강력한 성능<br>
단점: 계산 복잡성 높음<br>
모델식: LibSVM 기반 배깅<br>
응용분야: 고차원 데이터 분석<br>

### 8-16. Bagging Logistic w
특성: 로지스틱 회귀 기반 배깅<br>
장점: 빠르고 효율적<br>
단점: 과적합 위험<br>
모델식: 로지스틱 회귀 기반 배깅<br>
응용분야: 회귀 문제<br>

### 8-17. Bagging LWL w
특성: 지역 가중치 학습(Locally Weighted Learning)을 사용하는 배깅<br>
장점: 데이터의 지역적 패턴을 잘 포착<br>
단점: 계산 비용 높음<br>
모델식: LWL 기반 배깅<br>
응용분야: 실시간 데이터 처리<br>

### 8-18. Bagging MultilayerPerceptron w
특성: MLP 신경망 기반 배깅<br>
장점: 비선형 데이터 처리에 강점<br>
단점: 훈련 시간이 길 수 있음<br>
모델식: MLP 기반 배깅<br>
응용분야: 이미지 및 패턴 인식<br>

### 8-19. Bagging NaiveBayes w
특성: 나이브 베이즈 기반 배깅<br>
장점: 빠르고 간단함<br>
단점: 데이터 독립 가정에 따른 성능 변동<br>
모델식: NaiveBayes 기반 배깅<br>
응용분야: 텍스트 및 간단한 데이터 분류<br>

### 8-20. Bagging OneR w
특성: OneR 규칙 기반 배깅<br>
장점: 매우 간단하고 빠름<br>
단점: 복잡한 문제에 적합하지 않음<br>
모델식: OneR 기반 배깅<br>
응용분야: 간단한 분류 문제<br>

### 8-21. Bagging PART w
특성: PART 규칙 기반 배깅<br>
장점: 규칙 기반 학습<br>
단점: 계산 복잡성<br>
모델식: PART 기반 배깅<br>
응용분야: 텍스트 분류<br>

### 8-22. Bagging RandomForest w
특성: 랜덤 포레스트 기반 배깅<br>
장점: 높은 정확도<br>
단점: 계산 복잡성<br>
모델식: 랜덤 포레스트 기반 배깅<br>
응용분야: 고차원 데이터 분류<br>

### 8-23. Bagging RandomTree w
특성: 랜덤 트리 기반 배깅<br>
장점: 다양한 트리 구조 생성 가능<br>
단점: 성능 변동이 큼<br>
모델식: RandomTree 기반 배깅<br>
응용분야: 복잡한 분류 문제<br>

### 8-24. Bagging REPTree w
특성: REPTree 기반 배깅<br>
장점: 가지치기 기능으로 과적합 방지<br>
단점: 계산 비용이 높음<br>
모델식: REPTree 기반 배깅<br>
응용분야: 텍스트 분류, 다양한 데이터 분석<br>
<br>

## 9. Stacking (STC)
### 9-1. Stacking w
특성: ZeroR을 메타 분류기 및 기본 분류기로 사용하는 스태킹 앙상블<br>
장점: 간단한 구조로 빠른 학습 속도<br>
단점: 단순한 분류기 사용으로 인해 복잡한 문제에서는 성능 저하 가능<br>
모델식: Stacking 앙상블 (Wolpert, 1992)<br>
응용분야: 기본적인 분류 문제, 예비 분석<br>

### 9-2. StackingC w
특성: Seewald(2002)에 따른 더 효율적인 스태킹 앙상블로, 선형 회귀를 메타 분류기로 사용<br>
장점: 메타 분류기로 선형 회귀를 사용하여 예측 성능 향상<br>
단점: 특정한 데이터 구조에 의존할 수 있음<br>
모델식: 스태킹 앙상블 (Seewald, 2002)<br>
응용분야: 복잡한 회귀 문제, 데이터 마이닝<br>
<br>

## 10. Random Forests (RF)
### 10-1. rforest R
특성: Breiman(2001)의 랜덤 포레스트 앙상블을 R의 randomForest 함수로 생성, 트리 수(ntree)=500, 입력 변수 수(mtry)=√#inputs 사용<br>
장점: 높은 정확도 및 강건한 성능<br>
단점: 메모리 소비가 많음<br>
모델식: 랜덤 포레스트 앙상블<br>
응용분야: 다양한 분류 문제<br>

### 10-2. rf t
특성: caret 패키지를 통해 randomForest 함수 사용, 트리 수(ntree)=500, mtry=2:3:29로 파라미터 조정<br>
장점: 간편한 파라미터 튜닝<br>
단점: 파라미터에 따라 성능이 크게 변동 가능<br>
모델식: 랜덤 포레스트 앙상블<br>
응용분야: 데이터 마이닝 및 예측 모델링<br>

### 10-3. RRF t
특성: 정규화된 랜덤 포레스트, caret 패키지를 통한 RRF 함수 사용, mtry=2, 조정 파라미터 coefReg={0.01, 0.5, 1}<br>
장점: 정규화 기능으로 성능 향상<br>
단점: 추가적인 파라미터 조정 필요<br>
모델식: 정규화 랜덤 포레스트<br>
응용분야: 고차원 데이터 분석<br>

### 10-4. cforest t
특성: 조건부 추론 트리(CTree)의 랜덤 포레스트 및 배깅 앙상블, 각 CTree에서 평균 관측 가중치를 사용<br>
장점: 통계적 유의성을 반영<br>
단점: 복잡한 모델 구조<br>
모델식: 조건부 추론 트리 기반<br>
응용분야: 생물정보학 및 의료 데이터 분석<br>

### 10-5. parRF t
특성: 병렬 랜덤 포레스트 구현, randomForest 패키지 사용, mtry=2:2:8<br>
장점: 빠른 훈련 시간<br>
단점: 계산 복잡성 증가<br>
모델식: 병렬 랜덤 포레스트<br>
응용분야: 실시간 데이터 처리<br>

### 10-6. RRFglobal t
특성: RRF 패키지를 사용하여 RRF 생성, mtry=2, coefReg=0.01:0.12:1 조정<br>
장점: 정규화로 인한 성능 향상<br>
단점: 특정 데이터에 의존할 수 있음<br>
모델식: 전역 정규화 랜덤 포레스트<br>
응용분야: 고차원 데이터 분석<br>

### 10-7. RandomForest w
특성: 500개의 랜덤 트리 기반 분류기 구현, 입력 수를 log(#inputs + 1)로 설정, 무제한 깊이<br>
장점: 높은 정확도 및 유연성<br>
단점: 메모리 소비가 높음<br>
모델식: 랜덤 포레스트<br>
응용분야: 이미지 처리, 텍스트 분류<br>

### 10-8. RotationForest w
특성: J48을 기본 분류기로 사용, 주성분 분석 필터와 3개의 입력 그룹, 잎당 2 패턴, 가지치기 신뢰도 C=0.25<br>
장점: 데이터 변형에 강함<br>
단점: 구현 복잡성<br>
모델식: 회전 랜덤 포레스트<br>
응용분야: 데이터 마이닝 및 복잡한 문제 해결<br>
<br>

## 11. Other Ensembles (OEN)
### 11-1. RandomCommittee w
특성: 다양한 시드를 사용하여 구축된 RandomTrees의 앙상블로, 각 기본 분류기의 출력 평균 사용<br>
장점: 다양한 모델의 평균을 통해 강건성 향상<br>
단점: 계산 복잡성<br>
모델식: 랜덤 위원회 앙상블<br>
응용분야: 분류 문제, 데이터 마이닝<br>

### 11-2. OrdinalClassClassifier w
특성: 순서형 분류 문제를 위한 앙상블 방법, J48 기본 분류기 사용, 신뢰도 임계값 C=0.25<br>
장점: 순서형 문제에 적합한 처리<br>
단점: 설정 복잡성<br>
모델식: 순서형 분류기 앙상블<br>
응용분야: 순서형 데이터 분석<br>

### 11-3. MultiScheme w
특성: 여러 ZeroR 분류기 중에서 교차 검증을 통해 선택하는 앙상블<br>
장점: 다양한 모델 중 최적의 선택 가능<br>
단점: 데이터에 따라 성능 변동 가능<br>
모델식: 다중 기법 앙상블<br>
응용분야: 다양한 분류 문제<br>

### 11-4. MultiClassClassifier w
특성: 이진 로지스틱 회귀 기반 기본 분류기를 사용하여 다중 클래스 문제 해결, One-Against-All 접근법 사용<br>
장점: 다중 클래스 문제에서 높은 성능<br>
단점: 모델의 복잡성 증가<br>
모델식: 다중 클래스 로지스틱 회귀<br>
응용분야: 다중 클래스 분류<br>

### 11-5. CostSensitiveClassifier w
특성: 오류 유형에 따라 가중치가 부여된 ZeroR 기본 분류기를 사용<br>
장점: 비용 민감한 오류 처리 가능<br>
단점: 가중치 설정에 따라 성능이 변동 가능<br>
모델식: 비용 민감 분류기<br>
응용분야: 비용 민감한 분류 문제<br>

### 11-6. Grading w
특성: "등급" ZeroR 기본 분류기를 사용하는 Grading 앙상블<br>
장점: 간단한 구조로 빠른 학습<br>
단점: 단순한 규칙으로 복잡한 문제에 한계<br>
모델식: 등급 기반 분류기<br>
응용분야: 기본적인 분류 문제<br>

### 11-7. END w
특성: 다중 클래스 데이터셋을 두 클래스 J48 트리 분류기로 분류하는 Nested Dichotomies 앙상블<br>
장점: 복잡한 문제를 단순한 이진 분류 문제로 변환 가능<br>
단점: 설정이 복잡할 수 있음<br>
모델식: 중첩 이분 분류기<br>
응용분야: 다중 클래스 분류<br>

### 11-8. Decorate w
특성: 고차원 다양성을 갖춘 15개의 J48 트리 분류기를 학습하는 앙상블<br>
장점: 높은 다양성으로 성능 향상<br>
단점: 인공 데이터 패턴에 의존<br>
모델식: 다수의 J48 기반 앙상블<br>
응용분야: 패턴 인식<br>

### 11-9. Vote w
특성: ZeroR 기본 분류기의 앙상블을 평균 규칙으로 결합<br>
장점: 간단한 규칙으로 조합 가능<br>
단점: 단순함으로 인해 복잡한 문제에 약함<br>
모델식: 투표 기반 앙상블<br>
응용분야: 다양한 분류 문제<br>

### 11-10. Dagging w
특성: SMO 기반 앙상블로, 훈련 데이터를 4개로 나누어 학습<br>
장점: 각 분류기의 성능 향상<br>
단점: 계산 비용 증가<br>
모델식: 다중 이분 분류기<br>
응용분야: 다중 클래스 문제<br>

### 11-11. LWL w
특성: 지역 가중치 학습 앙상블, DecisionStump 기본 분류기 사용<br>
장점: 지역 패턴을 잘 포착<br>
단점: 계산 복잡성<br>
모델식: 지역 가중치 기반<br>
응용분야: 패턴 인식<br>
<br>

## 12. Generalized Linear Models (GLM)
### 12-1. glm R
특성: R의 stats 패키지에 있는 glm 함수를 사용하여 이항 및 포아송 분포 기반의 GLM 구현<br>
장점: 다양한 문제에 적용 가능하며 해석이 용이<br>
단점: 분포 가정이 잘못될 경우 성능 저하<br>
모델식: GLM = Y ~ X (이항 또는 포아송 분포)<br>
응용분야: 이진 및 다중 클래스 분류<br>

### 12-2. glmnet R
특성: Lasso 또는 Elastic Net 정규화를 사용하는 GLM 훈련, glmnet 패키지의 glmnet 함수 이용<br>
장점: 모델 복잡도를 제어할 수 있어 과적합 방지<br>
단점: 정규화 파라미터 조정 필요<br>
모델식: GLM = Y ~ X (정규화 포함)<br>
응용분야: 이진 및 다중 클래스 문제, 고차원 데이터<br>

### 12-3. mlm R
특성: nnet 패키지의 multinom 함수를 사용하여 다중 로지스틱 회귀 모델 구축<br>
장점: MLP 신경망을 통해 비선형 관계 모델링 가능<br>
단점: 데이터에 따라 성능이 변동 가능<br>
모델식: 다중 로지스틱 회귀 모델<br>
응용분야: 다중 클래스 분류<br>

### 12-4. bayesglm t
특성: arm 패키지의 bayesglm 함수를 사용하여 베이지안 GLM 구현<br>
장점: 사전 확률을 고려한 유연한 모델링<br>
단점: 계산 비용이 높을 수 있음<br>
모델식: GLM = Y ~ X (베이지안 함수 기반)<br>
응용분야: 회귀 분석, 분류 문제<br>

### 12-5. glmStepAIC t
특성: Akaike 정보 기준(AIC)을 이용한 모델 선택 수행, MASS 패키지의 stepAIC 함수 사용<br>
장점: 자동으로 최적 모델 선택 가능<br>
단점: 잘못된 초기 모델 선택 시 성능 저하<br>
모델식: AIC 기반 GLM 모델<br>
응용분야: 변수 선택, 모델 최적화<br>
<br>

## 13. Nearest Neighbors (NN)
### 13-1. knn R
특성: class 패키지의 knn 함수를 사용하여 K-최근접 이웃(KNN) 구현, 이웃의 수를 1에서 37까지 조정<br>
장점: 간단하고 이해하기 쉬움<br>
단점: 고차원 데이터에서는 성능 저하<br>
모델식: $KNN = Class(x) = argmax(count(class_i))$<br>
응용분야: 분류 문제, 패턴 인식<br>

### 13-2. knn t
특성: caret 패키지의 knn 함수를 사용하여 KNN 구현, 이웃의 수를 5에서 23까지 조정 (10으로 설정)<br>
장점: 파라미터 조정이 용이<br>
단점: 대규모 데이터셋에서 느림<br>
모델식: $KNN = Class(x) = argmax(count(class_i))$<br>
응용분야: 분류 문제, 추천 시스템<br>

### 13-3. NNge w
특성: 비중첩 일반화 예제(classifier)인 NNge, 상호 정보 계산을 위한 하나의 폴더 사용<br>
장점: 일반화 성능 향상<br>
단점: 설정이 복잡할 수 있음<br>
모델식: NNge = Generalized exemplar classifier<br>
응용분야: 다양한 패턴 인식 문제<br>

### 13-4. IBk w
특성: KNN 분류기로, K를 교차 검증을 통해 조정, 선형 이웃 검색 및 유클리드 거리 사용<br>
장점: 성능이 뛰어나고 유연함<br>
단점: 계산 비용이 높음<br>
모델식: $KNN = Class(x) = argmax(count(class_i))$<br>
응용분야: 다양한 분류 문제, 데이터 마이닝<br>

### 13-5. IB1 w
특성: 단순한 1-NN 분류기<br>
장점: 구현이 간단하고 효율적<br>
단점: 과적합 가능성이 있음<br>
모델식: $1-NN = Class(x) = argmax(count(class_i))$<br>
응용분야: 기본적인 분류 문제, 패턴 인식<br>
<br>

## 14. Partial Least Squares (PLSR)
### 14-1. pls t
특성: pls 패키지의 mvr 함수를 사용하여 PLSR 모델을 적합, 구성 요소 수를 1에서 10까지 조정<br>
장점: 다양한 데이터 구조에 대해 효과적<br>
단점: 최적의 구성 요소 수 선택에 따른 성능 변동 가능<br>
모델식: PLSR = Y ~ X (구성 요소 수 조정)<br>
응용분야: 회귀 분석, 다변량 데이터 분석<br>

### 14-2. gpls R
특성: gpls 패키지를 사용하여 일반화된 PLS 모델 구축<br>
장점: 유연한 모델링 가능<br>
단점: 모델 복잡성 증가<br>
모델식: 일반화된 PLS = Y ~ X<br>
응용분야: 복잡한 회귀 분석<br>

### 14-3. spls R
특성: spls 패키지의 spls 함수를 사용하여 희소 PLS 회귀 모델 적합, K와 η 매개변수 조정<br>
장점: 변수 선택 및 차원 축소 가능<br>
단점: 매개변수 조정의 복잡성<br>
모델식: 희소 PLS = Y ~ X (K, η 매개변수 조정)<br>
응용분야: 고차원 데이터 처리<br>

### 14-4. simpls R
특성: pls 패키지의 plsr 함수를 사용하여 SIMPLS 방법으로 PLSR 모델 적합<br>
장점: 계산 속도 향상<br>
단점: 모델의 복잡성<br>
모델식: SIMPLS PLSR = Y ~ X<br>
응용분야: 다변량 회귀 분석<br>

### 14-5. kernelpls R
특성: plsr 함수 사용, 커널 PLSR 방법 적용, 8개의 주성분 사용<br>
장점: 많은 입력에 대해 빠른 계산<br>
단점: 입력 수가 패턴 수보다 클 경우 계산 복잡성<br>
모델식: 커널 PLSR = Y ~ X<br>
응용분야: 고차원 데이터 분석<br>

### 14-6. widekernelpls R
특성: plsr 함수 사용, widekernelpls 방법으로 PLSR 모델 적합<br>
장점: 입력 수가 패턴 수보다 클 때 빠른 계산<br>
단점: 모델 복잡성 증가<br>
모델식: 넓은 커널 PLSR = Y ~ X<br>
응용분야: 고차원 데이터 처리<br>
<br>

## 15. Logistic and multinomial regression (LMR)
### 15-1. SimpleLogistic w
특성: 선형 로지스틱 회귀 모델을 학습, LogitBoost를 사용하여 기본 회귀 함수로 모델을 적합<br>
장점: 간단하고 해석하기 쉬움<br>
단점: 비선형 관계 모델링 한계<br>
모델식: 로지스틱 회귀 = $log(p/(1-p)) = β₀ + β₁X₁ + ... + βₖXₖ$<br>
응용분야: 이진 및 다중 클래스 분류<br>

### 15-2. Logistic w
특성: 다항 로지스틱 회귀 모델을 학습, 리지 추정기를 사용하여 모델 적합<br>
장점: 높은 차원의 데이터에 유연하게 적용 가능<br>
단점: 모델 복잡성으로 인해 과적합 위험<br>
모델식: 로지스틱 회귀 = $log(p_i/p_k) = β₀ + β₁X₁ + ... + βₖXₖ$ (다중 클래스)<br>
응용분야: 다중 클래스 문제 해결<br>

### 15-3. multinom t
특성: nnet 패키지의 multinom 함수를 사용하여 다항 로그 선형 모델을 학습<br>
장점: 신경망 기반으로 비선형 관계 모델링 가능<br>
단점: 파라미터 조정이 필요하여 모델이 복잡해질 수 있음<br>
모델식: 다항 로그 선형 회귀 = $log(p_i/p_k) = β₀ + β₁X₁ + ... + βₖXₖ$<br>
응용분야: 다중 클래스 분류 문제<br>
<br>

## 16. Multivariate Adaptive Regression Splines (MARS)
### 16-1. mars R
특성: mda 패키지의 mars 함수를 사용하여 MARS 모델을 적합<br>
장점: 비선형 관계를 효과적으로 모델링할 수 있음<br>
단점: 복잡한 데이터에 대해 과적합의 위험<br>
모델식: MARS 모델 = $β₀ + Σ(β_j(X_j - T_j)^⁺)$ (절단 함수 포함)<br>
응용분야: 회귀 분석, 데이터 마이닝<br>

### 16-2. gcvEarth t
특성: earth 패키지의 earth 함수를 사용하여 상호 작용 항이 없는 덧셈 MARS 모델 구축, 빠른 MARS 방법 적용<br>
장점: 빠른 계산 속도와 단순한 해석<br>
단점: 상호 작용 항을 고려하지 않아 일부 문제에서 성능 저하 가능<br>
모델식: 덧셈 MARS = $β₀ + Σ(β_j(X_j - T_j)^⁺)$ (절단 함수 포함)<br>
응용분야: 다변량 회귀 분석, 데이터 예측<br>
<br>

## 17. Other Methods (OM)
### 17-1. pam t
특성: PAM(Partitioning Around Medoids) 알고리즘을 사용하여 중심 기반 분류기 구현<br>
장점: 잡음과 이상치에 강함<br>
단점: 계산 비용이 높음<br>
모델식: PAM 알고리즘 기반<br>
응용분야: 클러스터링 및 패턴 인식<br>

### 17-2. VFI w
특성: 특성 간의 간격을 기반으로 한 분류, VFI(Voting Feature Intervals) 사용<br>
장점: 간단하고 직관적인 해석 가능<br>
단점: 특정 데이터 유형에 의존할 수 있음<br>
모델식: VFI 기반 분류<br>
응용분야: 텍스트 분류 및 추천 시스템<br>

### 17-3. HyperPipes w
특성: 각 클래스의 패턴 범위에 따라 테스트 패턴을 분류<br>
장점: 빠르고 간단한 계산<br>
단점: 복잡한 데이터에 대한 성능 저하 가능<br>
모델식: 하이퍼파이프 기반 분류<br>
응용분야: 기본적인 분류 문제<br>

### 17-4. FilteredClassifier w
특성: 특정 데이터에 대해 필터링하여 J48 결정 트리를 사용하여 학습<br>
장점: 필터링된 데이터를 사용해 성능 향상<br>
단점: 필터링 과정에서 정보 손실 가능성<br>
모델식: 필터링된 J48 기반 분류<br>
응용분야: 텍스트 및 이미지 분류<br>

### 17-5. CVParameterSelection w
특성: 10겹 교차 검증을 통해 ZeroR의 최적 매개변수 선택<br>
장점: 성능 최적화 가능<br>
단점: 실행 시간이 길어질 수 있음<br>
모델식: 교차 검증 기반 매개변수 선택<br>
응용분야: 분류 문제 최적화<br>

### 17-6. ClassificationViaClustering w
특성: K-평균 클러스터링을 통해 데이터를 클러스터링한 후 분류<br>
장점: 클러스터링을 통해 복잡한 패턴을 포착 가능<br>
단점: 데이터의 초기 클러스터 수에 민감함<br>
모델식: 클러스터링 기반 분류<br>
응용분야: 대규모 데이터 분석<br>

### 17-7. AttributeSelectedClassifier w
특성: 중요 속성만을 선택하여 J48 결정 트리로 학습<br>
장점: 성능 향상 및 해석 용이<br>
단점: 잘못된 속성 선택 시 성능 저하<br>
모델식: 속성 선택 기반 J48<br>
응용분야: 데이터 마이닝<br>

### 17-8. ClassificationViaRegression w
특성: 클래스를 이진화하여 M5P 회귀 모델로 분류<br>
장점: 회귀 모델링을 통한 유연한 분류 가능<br>
단점: 설정이 복잡할 수 있음<br>
모델식: 회귀 기반 분류<br>
응용분야: 복잡한 회귀 및 분류 문제<br>

### 17-9. KStar w
특성: K-최근접 이웃(KNN) 방식으로 인스턴스 기반 분류<br>
장점: 유사성에 기반한 분류로 정확도 향상<br>
단점: 계산 비용이 높음<br>
모델식: KNN 기반 분류<br>
응용분야: 패턴 인식 및 데이터 분석<br>

### 17-10. gaussprRadial t
특성: kernlab 패키지의 가우스 과정 회귀를 사용하여 분류<br>
장점: 높은 정확도 제공<br>
단점: 대규모 데이터셋에서 느릴 수 있음<br>
모델식: 가우스 프로세스 기반 분류<br>
응용분야: 고차원 데이터 분석<br>
<br>


