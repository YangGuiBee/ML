#  02 : ML의 정의와 절차, 학습 방법과 모델

---
	 ▣ 2024년 10대 전략기술 트렌트
	 ▣ AI의 역사
	 ▣ AI의 정의
	 ▣ AI의 정의(최근 IT기업)
	 ▣ AI의 유형
	 ▣ AI의 활용 분야
  	 ▣ 실습 준비
---

## ▣ ML의 정의


### □ 1959년, Arthur Samuel (아서 새뮤얼)
정의 : "컴퓨터에 명시적으로 프로그래밍되지 않고도 학습할 수 있는 능력을 부여하는 연구 분야."<br>
표현 : "Field of study that gives computers the ability to learn without being explicitly programmed."<br>
설명 : 머신러닝이 기존의 명시적 프로그래밍 방식과 어떻게 다른지를 설명하면서 컴퓨터가 데이터를 통해 스스로 학습할 수 있는 능력을 강조<br>


### □ 1983년, Herbert A. Simon (허버트 사이먼)
정의 : "학습은 시스템이 주어진 작업에 대해 이전보다 더 나은 성능을 보일 때 발생한다."<br>
표현 : "Learning is any process by which a system improves performance from experience."<br>
설명 : 경험을 통한 성능 향상을 학습의 핵심으로 보았습니다. 이 정의는 특히 시스템이 경험을 통해 지속적으로 개선되는 과정을 강조<br>


### □ 1997년, Tom M. Mitchell (톰 미첼) 
정의 : "컴퓨터 프로그램이 경험 EEE에서 학습하며, 작업 TTT와 성능 측정 PPP와 관련하여 성능이 향상되었다면, 그 프로그램은 작업 TTT에 대해 경험 EEE로부터 학습한 것."<br>
표현 : "A computer program is said to learn from experience EEE with respect to some task TTT and performance measure PPP, if its performance on TTT, as measured by PPP, improves with experience EEE."<br>
설명 : "Machine Learning" 저서에서 학습의 세 가지 주요 요소(작업, 경험, 성능 측정)를 통해 머신러닝의 핵심 개념을 체계적으로 설명<br>


### □ 2004년, Ethem Alpaydin (에텀 알파이딘)
정의 : "데이터에서 패턴을 찾고, 이를 바탕으로 예측을 수행할 수 있는 알고리즘의 설계와 연구."
표현 : "Machine learning is the study of algorithms that learn from data and make predictions."
설명 : "Introduction to Machine Learning" 저서에서 머신러닝의 예측 기능에 중점을 두며, 데이터에서 패턴을 발견하고 이를 기반으로 예측하는 과정의 중요성을 강조


### □ 2008년, Andrew Ng (앤드류 응)
정의 : "머신러닝은 명시적으로 프로그래밍하지 않고 컴퓨터가 행동하는 방식을 학습하는 학문이다."<br>
표현 : "Machine learning is the field of study that enables computers to learn from data without being explicitly programmed."<br>
설명 : "Stanford Machine Learning" 강의에서 데이터 기반 학습과 자율 학습 능력의 중요성을 강조하며, 현대의 대규모 데이터와 복잡한 문제를 해결하는 머신러닝의 필요성을 반영<br>


### □  2012년, Kevin P. Murphy (케빈 머피)
정의 : "머신러닝은 데이터를 사용하여 예측 모델을 학습하는 데 중점을 둔 컴퓨터 과학의 하위 분야이다."<br>
표현 : "Machine learning is a subfield of computer science that focuses on the development of algorithms that can learn from and make predictions on data."<br>
설명 : "Machine Learning: A Probabilistic Perspective" 저서에서 머신러닝의 이론적 연구와 실질적인 예측 모델의 개발에 대한 중요성을 강조<br>



머신 러닝 교과서 2판

##  2장: 간단한 분류 알고리즘 훈련

### 목차

- 인공 뉴런: 초기 머신 러닝의 간단한 역사
    - 인공 뉴런의 수학적 정의
    - 퍼셉트론 학습 규칙
- 파이썬으로 퍼셉트론 학습 알고리즘 구현
    - 객체 지향 퍼셉트론 API
    - 붓꽃 데이터셋에서 퍼셉트론 훈련
- 적응형 선형 뉴런과 학습의 수렴
    - 경사 하강법으로 비용 함수 최소화
    - 파이썬으로 아달린 구현
    - 특성 스케일을 조정하여 경사 하강법 결과 향상
    - 대규모 머신 러닝과 확률적 경사 하강법
- 요약

### 코드 사용 방법 안내

이 책의 코드를 사용하는 가장 좋은 방법은 주피터 노트북(`.ipynb` 파일)입니다. 주피터 노트북을 사용하면 단계적으로 코드를 실행하고 하나의 문서에 편리하게 (그림과 이미지를 포함해) 모든 출력을 저장할 수 있습니다.

![](images/jupyter-example-1.png)

주피터 노트북은 매우 간단하게 설치할 수 있습니다. 아나콘다 파이썬 배포판을 사용한다면 터미널에서 다음 명령을 실행하여 주피터 노트북을 설치할 수 있습니다:

    conda install jupyter notebook

다음 명령으로 주피터 노트북을 실행합니다.

    jupyter notebook

브라우저에서 윈도우가 열리면 원하는 `.ipynb`가 들어 있는 디렉토리로 이동할 수 있습니다.

**설치와 설정에 관한 더 자세한 내용은 1장의 [README.md 파일](../ch01/README.md)에 있습니다.**

**(주피터 노트북을 설치하지 않았더라도 깃허브에서 [`ch02.ipynb`](https://github.com/rickiepark/python-machine-learning-book-3rd-edition/blob/master/ch02/ch02.ipynb)을 클릭해 노트북 파일을 볼 수 있습니다.)**.

코드 예제 외에도 주피터 노트북에는 책의 내용에 맞는 섹션 제목을 함께 실었습니다. 또한 주피터 노트북에 원본 이미지와 그림을 포함시켰기 때문에 책을 읽으면서 코드를 쉽게 따라할 수 있으면 좋겠습니다.

![](images/jupyter-example-2.png)
