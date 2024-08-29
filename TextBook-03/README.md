#  03 : Data Collection, Data Processing : Python, 라이브러리(NumPy, Pandas, Scikit-Learn, MNIST, TensorFlow...)
---
	 ▣ 데이터 수집
	 ▣ 데이터 전처리	 
	 ▣ 데이터셋 분리  	  
  	 ▣ Python 라이브러리	
---

## ▣ 데이터 수집

데이터 수집은 머신러닝 절차(분석 문제 정의 → 데이터 수집 → 탐색적 데이터 분석(EDA) → 피처 엔지니어링 → 예측 모델 개발 → 서비스 적용) 중 분석 문제 정의 다음의 단계이며, 이 단계에서는 정의한 문제를 해결하기 위한 데이터들을 수집하는 단계이다. 어떤 데이터를 수집하느냐에 따라 문제 해결을 위한 접근 방식이 달라지며, 이것은 데이터의 유형도 신경써야할 필요가 있다. 머신러닝 프로젝트에서 두 번째 단계인 '데이터 수집'은 분석의 기반이 되는 데이터를 확보하는 과정으로 다음과 같은 5가지 단계로 이루어진다.<br>

**(1) 원시 데이터 수집 :** 중요도에 비해 관심을 덜 받는 경우가 많지만 데이터 수집은 우수한 알고리즘을 얻기 위한 첫 걸음이다. 일반적으로 대량의 데이터 집합을 사용하는 단순한 모델이 소량의 데이터 집합을 사용하는 진보된 모델보다 성능이 더 우수하다. 
**(2) 데이터 마트 생성 :** 데이터 마트는 특정 주제나 부서에 초점을 맞춘 작은 규모의 데이터 웨어하우스를 의미하며, 이 단계에서는 필요한 데이터를 특정 주제나 목적에 맞게 분류하거나 구성한다. 이를 통해 필요한 데이터를 효율적으로 관리하고 사용할 수 있다. 예컨대, '고객 만족도'라는 주제는 고객 ID, 구매 이력, 제품 리뷰, 고객 서비스 이력 등을 포함하는 데이터 마트를 생성한다.<br>
**(3) 데이터 정합성 평가 :** 수집된 데이터의 질을 평가하는 과정으로 데이터의 정확성, 일관성, 완전성, 신뢰성 등을 검토하고, 이상치나 결측치, 중복 값 등이 있는지 확인한다. 이를 통해 데이터의 정합성을 보장하고, 분석의 신뢰성을 높일 수 있다. 예컨대, 고객 ID의 중복, 제품 리뷰의 결측치, 구매 이력의 이상치 등을 확인하고, 이를 수정하거나 제거하여 데이터의 정합성을 보장한다. <br>
**(4) 데이터 취합 :** 여러 출처에서 수집된 데이터를 하나의 데이터 세트로 합치는 과정으로 동일한 개체나 사건을 나타내는 데이터가 일관된 방식으로 표현되고 연결되어야야만 이를 통해 통합된 정보를 제공하고, 분석의 효율성을 높일 수 있다. 예컨대, 구매 이력 데이터, 제품 리뷰 데이터, 고객 서비스 이력 데이터 등을 고객 ID를 기준으로 합친다.<br>
**(5) 데이터 표준화 :** 서로 다른 소스에서 수집된 데이터는 종종 다른 형식이나 구조로 저장되어 있는 경우는 단계에서는 모든 데이터를 일관된 포맷으로 변환하여, 분석이나 처리가 쉽도록 한다.예컨대, 일자 데이터가 '년-월-일' 형식으로 저장된 곳도 있고, '월/일/년' 형식으로 저장된 곳도 있다면, 이를 일관된 형식으로 통일시킨다. “Garbage in, garbage out”라는 말이 있듯이, 무의미한 데이터가 들어오면 나가는 데이터도 무의미하다. 데이터의 출처에 따라 부가적인 포맷과 표준화가 필요할 수 있으며, 고품질의 대량의 데이터 집합이라도 해도 데이터 포맷이 잘못되면 힘을 발휘할 수 없다. 이 점은 여러 출처의 데이터를 집계할 때 특히 유의해야 한다.<br>

이렇게 데이터 수집 단계를 통해 필요한 데이터를 효과적으로 확보하고, 그 데이터의 질을 보장하고, 데이터를 적절하게 관리하고 사용할 수 있으며, 이 단계를 잘 수행하면, 그 이후의 분석 과정에서 좀 더 정확하고 효율적인 결과를 얻을 수 있다다. 이렇게 데이터 수집 단계를 거친 후에는, 데이터의 질을 보장하고, 필요한 정보를 효율적으로 제공하는 '탐색적 데이터 분석(EDA)', '피처 엔지니어링', '예측 모델 개발', '서비스 적용' 등의 작업을 진행할 수 있다.<br>

머신러닝 데이터셋(dataset) 사이트 40가지 모음 : https://kr.appen.com/blog/best-datasets/

<br><br>

## ▣ 데이터 전처리
수집된 데이터는 컴퓨팅 학습을 위해서 레이블 인코딩(Label Encoding)과 원핫 인코딩(One-hot Encoding)을 통해 범주형 데이터를 수치형 데이터로 변환이 필요하다.<br>
**레이블 인코딩 :** 각 카테고리를 숫자로 대응시켜서 변환한다. 예컨대, "red", "green", "blue"라는 3개의 카테고리가 있다면 "red"를 1로, "green"을 2로, "blue"를 3으로 변환하는 것이다. 이 방법은 간단하고 직관적이지만, 각 카테고리가 가지는 값의 크기 차이가 있을 경우 예측 결과에 영향을 미칠 수 있다.<br>
**원핫 인코딩 :** 각 카테고리를 벡터 형태로 변환한다. 예컨대, "red", "green", "blue"라는 3개의 카테고리가 있다면 "red"는 [1, 0, 0], "green"은 [0, 1, 0], "blue"는 [0, 0, 1]로 변환하는 것이다. 이 방법은 각 카테고리를 독립적인 변수로 취급하기 때문에 각 카테고리가 가지는 값의 크기 차이를 고려하지 않기 때문에 범주형 변수의 카테고리가 많을수록 차원이 커지는 단점이 있지만, 예측 결과에 영향을 미치는 위험이 적다.<br>
따라서, 레이블 인코딩은 카테고리가 서열을 가지는 경우(예: "bad", "average", "good")나 카테고리의 수가 적을 경우에 사용하고, 원핫 인코딩은 카테고리의 수가 많을 경우에 사용한다.<br>

https://davinci-ai.tistory.com/15
https://ysyblog.tistory.com/71

<br><br><br>

## ▣ 데이터셋 분리
데이터를 적절히 포맷했고 품질을 확인했고 모든 관련 특성을 확보했다면, 이제 머신러닝을 위해 데이터 분할이 필요하다. 분할의 단계는 총 3단계로 나누어 볼 수 있다.<br>
![](./images/DataSet.PNG)
<br>
1 단계 : training과 test데이터로 분리한다.<br>
2 단계 : training데이터를 training과 validation으로 분리한다.<br>
3 단계 : training데이터로 모델을 만들고 validation데이터로 검증한다. 만족시 해당모델을 train과 validation데이터를 합쳐서 학습을 시킨후 test데이터를 넣어 확인한다.<br>

데이터 셋의 비중은 정해진 것은 아니나 보통 Training set : Validation set : Test sets = 60 : 20 : 20 으로 설정한다.<br>
**훈련 데이터(Training set) :** 모델을 학습하는데 사용된다. Training set으로 모델을 만든 뒤 동일한 데이터로 성능을 평가해보기도 하지만, 이는 cheating이 되기 때문에 유효한 평가는 아니다. 마치 모의고사와 동일한 수능 문제지를 만들어 대입 점수를 매기는 것과 같다. Training set은 Test set이 아닌 나머지 데이터 set을 의미하기도 하며, Training set 내에서 또 다시 쪼갠 Validation set이 아닌 나머지 데이터 set을 의미하기도 한다. 문맥상 Test set과 구분하기 위해 사용되는지, Validation과 구분하기 위해 사용되는지를 확인해야 한다.<br>
**검정 데이터(Validation set) :** training set으로 만들어진 모델의 성능을 측정하기 위해 사용된다. 일반적으로 어떤 모델이 가장 데이터에 적합한지 찾아내기 위해서 다양한 파라미터와 모델을 사용해보게 되며, 그 중 validation set으로 가장 성능이 좋았던 모델을 선택한다. 테스트 데이터(Test set)은 validation set으로 사용할 모델이 결정 된 후, 마지막으로 딱 한번 해당 모델의 예상되는 성능을 측정하기 위해 사용된다. 이미 validation set은 여러 모델에 반복적으로 사용되었고 그중 운 좋게 성능이 보다 더 뛰어난 것으로 측정되어 모델이 선택되었을 가능성이 있다. 때문에 이러한 오차를 줄이기 위해 한 번도 사용해본 적 없는 test set을 사용하여 최종 모델의 성능을 측정하게 된다.<br>
**Validation set :** 여러 모델들 각각에 적용되어 성능을 측정하며, 최종 모델을 선정하기 위해 사용된다. 반면 test set은 최종 모델에 대해 단 한번 성능을 측정하며, 앞으로 기대되는 성능을 예측하기 위해 사용된다. Training set으로 모델들을 만든 뒤, validation set으로 최종 모델을 선택하게 된다. 최종 모델의 예상되는 성능을 보기 위해 test set을 사용하여 마지막으로 성능을 평가한다. 그 뒤 실제 사용하기 전에는 쪼개서 사용하였던 training set, validation set, test set 을 모두 합쳐 다시 모델을 training 하여 최종 모델을 만든다. 기존 training set만을 사용하였던 모델의 파라미터와 구조는 그대로 사용하지만, 전체 데이터를 사용하여 다시 학습시킴으로써 모델이 조금 더 튜닝되도록 만든다. 혹은 data modeling을 진행하는 동안 새로운 데이터를 계속 축적하는 방법도 있다. 최종 모델이 결정되었을 때 새로 축적된 data를 test data로 사용하여 성능평가를 할 수도 있다.<br>

![](./images/DataSet2.PNG)

	from sklearn.model_selection import train_test_split
	
	# train-test분리
	X_train, X_test, y_train, y_test = train_test_split(df['feature'],df['target'])
	
	# train-validation분리
	X2_train, X2_val, y2_train, y_val = train_test_split(X_train, y_train)


<br><br><br>

https://velog.io/@ljs7463/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%95%99%EC%8A%B5%EC%9D%84-%EC%9C%84%ED%95%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EB%B6%84%EB%A6%ACtraintestvalidation


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
