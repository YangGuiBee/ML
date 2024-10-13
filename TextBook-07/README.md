#  07 : CASE Study

---

  https://www.dbpia.co.kr/
  
  https://www.riss.kr/

  https://news.hada.io/search?q=ML
	
---  

## Do we need hundreds of classifiers to solve real world classification problems?

연구 목적 : 이 연구는 다양한 머신 러닝 분류 알고리즘들의 성능을 비교하여 특정 데이터셋에 대해 가장 성능이 좋은 분류기를 찾는 것을 목표로 하였다.<br> 

연구 방법론 : 179개의 분류기를 17개의 패밀리로 나누어 R, Weka, C, Matlab 등 다양한 소프트웨어 도구를 통해 분석하였다. 연구에 사용된 데이터셋은 UCI 머신러닝 저장소와 실제 문제 데이터 121개를 50%의 데이터로 파라미터 튜닝을 수행한 후, 교차 검증을 통해 성능을 평가하는 방식으로 진행하였다.<br>

연구 대상: 분류 알고리즘
랜덤 포레스트(Random Forest)
서포트 벡터 머신(SVM, Gaussian 커널 포함)
신경망(Multi-layer Perceptron 포함)
부스팅(AdaBoost, LogitBoost 등)
베이지안 분류기 등

연구 결과:
랜덤 포레스트(Random Forest): 대부분의 데이터셋에서 가장 좋은 성능을 기록했다. 특히, R에서 구현된 랜덤 포레스트 알고리즘은 최대 정확도의 94.1%를 달성했으며, 121개의 데이터셋 중 84.3%에서 90% 이상의 정확도를 보였다.
SVM : 두 번째로 좋은 성능을 보였으며, Gaussian 커널을 사용한 SVM은 최대 정확도의 92.3%를 기록했습니다.
기타 모델: 신경망(MLP)과 부스팅(Boosting) 알고리즘도 높은 성능을 기록했습니다. 특히 C5.0, Extreme Learning Machine(ELM), 다층 퍼셉트론(MLP) 기반의 avNNet은 상위 성능 그룹에 포함되었습니다.
성능 변동: 모든 알고리즘이 데이터셋에 따라 성능이 다르게 나타났으며, No-Free-Lunch 정리에 따라 어떤 알고리즘도 모든 데이터셋에서 최고의 성능을 보이지 않았습니다.

결론 : 이 연구는 다수의 분류기를 비교함으로써 다양한 알고리즘 간의 성능 차이를 분석하고, 특정 알고리즘이 특정 상황에서 더 유리할 수 있음을 보여주었다. 연구 결과, 랜덤 포레스트와 SVM 계열의 알고리즘들이 전반적으로 우수한 성능을 보였으나, 각 데이터셋에 따라 최적의 알고리즘이 달라질 수 있음을 확인했다.<br>
