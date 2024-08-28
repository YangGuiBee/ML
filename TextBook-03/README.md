#  03 : Data Collection, Data Processing : Python, 라이브러리(NumPy, Pandas, Scikit-Learn, MNIST, TensorFlow...)
---
	 ▣ Python 라이브러리	
  	 ▣ 데이터 수집
	 ▣ 데이터 전처리
	 ▣ 범주형 데이터 처리
	 ▣ 훈련 데이터셋과 테스트 데이터셋 분리  	  
---

## ▣ Python 라이브러리

### 【NumPy】
행렬이나 일반적으로 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 파이썬의 라이브러리<br>
NumPy는 데이터 구조 외에도 수치 계산을 위해 효율적으로 구현된 기능을 제공<br>
NumPy 공식문서 : https://numpy.org/doc/stable/user/whatisnumpy.html

<br>

### 【Pandas】
데이터 조작 및 분석을 위한 파이썬 프로그래밍 언어 용으로 작성된 소프트웨어 라이브러리<br>
숫자 테이블과 시계열을 조작하기 위한 데이터 구조와 연산을 제공<br>
Pandas란 이름은 한 개인에 대해 여러 기간 동안 관찰을 한다는 데이터 세트에 대한 계량 경제학 용어인 "패널 데이터"에서 파생<br>
또한 "Python 데이터 분석"이라는 문구 자체에서 따온 것이기도 하다.<br>
Wes McKinney는 2007년부터 2010년까지 연구원으로 있을 때 AQR Capital에서 pandas를 만들기 시작했다.<br>
Pandas 공식문서 : https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html

<br>

### 【Scikit-Learn】
Scikit-learn(이전 명칭: scikits.learn, sklearn)은 파이썬 프로그래밍 언어용 자유 소프트웨어 기계 학습 라이브러리<br>
다양한 분류, 회귀, 그리고 서포트 벡터 머신, 랜덤 포레스트, 그라디언트 부스팅, k-평균, DBSCAN을 포함한 클러스터링 알고리즘<br>
파이썬의 수치 및 과학 라이브러리 NumPy 및 SciPy와 함께 운용되도록 설계<br>
Scikit-learn 공식문서 : https://scikit-learn.org/stable/user_guide.html
<br>
SciPy 공식문서 : https://docs.scipy.org/doc/scipy/

<br>

### 【MNIST】
MNIST(Modified National Institute of Standards and Technology database)는 손으로 쓴 숫자들로 이루어진 대형 데이터베이스<br>
다양한 화상처리시스템과 기계학습 분야의 트레이닝 및 테스트에 널리 사용<br>
MNIST 데이터베이스는 60,000개의 트레이닝 이미지와 10,000개의 테스트 이미지를 포함<br>
MNIST 사용가이드 : https://guide.ncloud-docs.com/docs/tensorflow-tensorflow-1-3

<br>

### 【TensorFlow】
머신러닝 및 인공 지능을 위한 무료 오픈소스 소프트웨어 라이브러리<br>
다양한 작업에 사용할 수 있지만 특히 심층 신경망의 교육 및 추론에 중점<br>
연구 및 생산에서 Google의 내부 사용을 위해 Google Brain 팀에서 개발<br>
TensorFlow는 Python, JavaScript, C++ 및 Java 등 다양한 프로그래밍 언어와 많은 분야의 다양한 애플리케이션에서 쉽게 사용가능<br>
TensorFlow 공식문서 : https://www.tensorflow.org/?hl=ko

<br>

### 【Matplotlib】
Python 프로그래밍 언어 및 수학적 확장 NumPy 라이브러리를 활용한 플로팅 라이브러리<br>
Tkinter , wxPython , Qt 또는 GTK 와 같은 범용 GUI 툴킷을 사용하여 애플리케이션에 플롯을 포함 하기 위한 객체 지향 API를 제공<br> 
Matplotlib 공식문서 : https://matplotlib.org/stable/
<br>
Matplotlib 가이드 : https://wikidocs.net/92071

<br><br><br>

## ▣ 데이터 수집
데이터 수집은 머신러닝 절차(분석 문제 정의 → 데이터 수집 → 탐색적 데이터 분석(EDA) → 피처 엔지니어링 → 예측 모델 개발 → 서비스 적용) 중 분석 문제 정의 다음의 단계이며, 이 단계에서는 정의한 문제를 해결하기 위한 데이터들을 수집하는 단게입니다. 어떤 데이터를 수집하느냐에 따라 문제 해결을 위한 접근 방식이 달라지며, 이것은 데이터의 유형도 신경써야할 필요가 있습니다. 머신러닝 프로젝트에서 두 번째 단계인 '데이터 수집'은 분석의 기반이 되는 데이터를 확보하는 과정입니다. 이 과정은 다음과 같은 4가지 단계로 이루어집니다.

1. 데이터 마트 생성: 데이터 마트는 특정 주제나 부서에 초점을 맞춘 작은 규모의 데이터 웨어하우스를 의미합니다. 이 단계에서는 필요한 데이터를 특정 주제나 목적에 맞게 분류하거나 구성합니다. 이를 통해 필요한 데이터를 효율적으로 관리하고 사용할 수 있습니다.
2. 데이터 정합성 평가: 수집된 데이터의 질을 평가하는 과정입니다. 데이터의 정확성, 일관성, 완전성, 신뢰성 등을 검토하고, 이상치나 결측치, 중복 값 등이 있는지 확인합니다. 이를 통해 데이터의 정합성을 보장하고, 분석의 신뢰성을 높일 수 있습니다.
3. 데이터 취합: 여러 출처에서 수집된 데이터를 하나의 데이터 세트로 합치는 과정입니다. 이 때, 동일한 개체나 사건을 나타내는 데이터가 일관된 방식으로 표현되고 연결되어야 합니다. 이를 통해 통합된 정보를 제공하고, 분석의 효율성을 높일 수 있습니다.
4. 데이터 포맷 통일: 서로 다른 소스에서 수집된 데이터는 종종 다른 형식이나 구조로 저장되어 있습니다. 이 단계에서는 모든 데이터를 일관된 포맷으로 변환하여, 분석이나 처리가 쉽도록 합니다.

이렇게 데이터 수집 단계를 통해 필요한 데이터를 효과적으로 확보하고, 그 데이터의 질을 보장하고, 데이터를 적절하게 관리하고 사용할 수 있습니다. 이 단계를 잘 수행하면, 그 이후의 분석 과정에서 좀 더 정확하고 효율적인 결과를 얻을 수 있습니다.


## ▣ 데이터 전처리
데이터 인코딩 : One-Hot encoding<br>
스케일링 : StandardScaler, MinMaxScaler<br>

## ▣ 범주형 데이터 처리

## ▣ 훈련 데이터셋과 테스트 데이터셋 분리




<br><br><br>

---

### 코드 사용 방법 안내

이 책의 코드를 사용하는 가장 좋은 방법은 주피터 노트북(`.ipynb` 파일)입니다. 주피터 노트북을 사용하면 단계적으로 코드를 실행하고 하나의 문서에 편리하게 (그림과 이미지를 포함해) 모든 출력을 저장할 수 있습니다.

![](../TextBook-02/images/jupyter-example-1.png)

주피터 노트북은 매우 간단하게 설치할 수 있습니다. 아나콘다 파이썬 배포판을 사용한다면 터미널에서 다음 명령을 실행하여 주피터 노트북을 설치할 수 있습니다:

    conda install jupyter notebook

다음 명령으로 주피터 노트북을 실행합니다.

    jupyter notebook

브라우저에서 윈도우가 열리면 원하는 `.ipynb`가 들어 있는 디렉토리로 이동할 수 있습니다.

**설치와 설정에 관한 더 자세한 내용은 1장의 [README.md 파일](../TextBook-01/README.md)에 있습니다.**

**(주피터 노트북을 설치하지 않았더라도 깃허브에서 [`ch03.ipynb`](https://github.com/rickiepark/python-machine-learning-book-3rd-edition/blob/master/ch03/ch03.ipynb)을 클릭해 노트북 파일을 볼 수 있습니다.)**.

코드 예제 외에도 주피터 노트북에는 책의 내용에 맞는 섹션 제목을 함께 실었습니다. 또한 주피터 노트북에 원본 이미지와 그림을 포함시켰기 때문에 책을 읽으면서 코드를 쉽게 따라할 수 있으면 좋겠습니다.

![](../TextBook-02/images/jupyter-example-2.png)
