


#  11 : 지도 학습(Supervised Learning, SL) : 회귀(regression) + 분류(classification)

---

	[1] 판별 분석 (Discriminant Analysis)
		[1-1] LDA (Linear Discriminant Analysis)
		[1-2] QDA (Quadratic Discriminant Analysis)
		
	[2] 차원축소 (Dimensionality Reduction)
		[2-1] PCR (Principal Component Regression) : PCA(비지도학습의 차원축소) + 회귀
		[2-2] PLS (Partial Least Squares)
		[2-3] PLS-DA (Partial Least Squares Discriminant Analysis)
		[2-4] Supervised PCA

	[3] 트리 기반 (Tree-based)
		[3-1] 결정 트리 (Decision Tree)
		[3-2] 랜덤 포레스트 (Random Forest)

	[4] 거리 기반 (Distance-based)
		[4-1] k-최근접 이웃 (k-Nearest Neighbors, K-NN)
		[4-2] 서포트 벡터 머신 (Support Vector Machine, SVM)
	
---  

# [1-1] Linear Discriminant Analysis (LDA)
▣ 정의: 데이터를 직선(또는 평면) 하나로 깔끔하게 나누는 방법으로<br>
데이터가 여러 그룹으로 나뉘어 있을 때, 그룹 사이의 차이는 최대화하면서 같은 그룹 안의 차이는 최소화하도록 데이터를 잘 구분해주는 선(혹은 초평면)을 찾는 방법(가정 : 모든 클래스의 공분산이 같다. 모양이 같은 1차식 곡선)<br>
▣ 필요성: 클래스 간 분리를 극대화하면서 데이터를 저차원으로 투영하여 분류 문제의 성능을 향상시키기 위해 필요<br>
▣ 장점: 클래스 분리를 극대화하여 분류 성능을 개선할 수 있으며, 선형 변환을 통해 효율적으로 차원을 축소<br>
▣ 단점: 데이터가 선형적으로 구분되지 않는 경우 성능이 저하될 수 있으며, 클래스 간 분포가 정규 분포를 따를 때 더 효과적<br>
▣ Scikit-learn 클래스명 : sklearn.discriminant_analysis.LinearDiscriminantAnalysis<br> 
▣ 가이드 : https://scikit-learn.org/stable/modules/lda_qda.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html<br>

![](./images/LDA_1.png)
<br>

**([1-1] LDA 예제 소스)**

	# ============================================
	#  4차원 Iris 데이터를 2차원으로 차원축소하여 클래스 간 분리가 최대가 되는 축 찾기
	# ============================================
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	import matplotlib.pyplot as plt
	from sklearn.datasets import load_iris

	# --------------------------------------------------
	# ① 데이터 로드
	# --------------------------------------------------
	data = load_iris()      # Iris 데이터셋 로드
	X = data.data           # 입력 특성 (150 × 4) : sepal length, sepal width, petal length, petal width
	y = data.target         # 출력 레이블 (150,) : 3개 클래스 (0=setosa, 1=versicolor, 2=virginica)

	# --------------------------------------------------
	# ② LDA 모델 생성 및 학습
	# --------------------------------------------------
	# Linear Discriminant Analysis:
	#   - 클래스 간 분산(S_b)을 크게 하고 클래스 내 분산(S_w)을 작게 하여
	#     Fisher의 판별 기준 (J = |S_b| / |S_w|) 을 최대화하는 투영축 찾기
	#   - C개의 클래스가 있으면 최대 C-1 차원으로 축소 가능
	#     (여기서는 3클래스 → 최대 2차원으로 축소 가능)
	lda = LinearDiscriminantAnalysis(n_components=2)

	# fit_transform():
	#   1) LDA 모델 학습 (fit)
	#   2) 데이터를 새로운 LDA 공간으로 투영 (transform)
	X_lda = lda.fit_transform(X, y)    # X_lda shape: (150, 2)

	# --------------------------------------------------
	# ③ 결과 시각화
	# --------------------------------------------------
	# LDA로 변환된 두 축(Component 1, 2)을 기준으로 산점도 작성
	# 각 점은 하나의 샘플, 색상(c=y)은 클래스(붓꽃 품종)를 나타냄
	plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
	plt.xlabel("LDA Component 1")
	plt.ylabel("LDA Component 2")
	plt.title("LDA on Iris Dataset")
	plt.colorbar(label="Class label (0=setosa, 1=versicolor, 2=virginica)")
	plt.show()

	# --------------------------------------------------
	# [그래프 해석]
	# --------------------------------------------------
	# 세 가지 품종(색상별로 구분)이 LDA 공간에서 서로 잘 분리되어 나타남
	# LDA Component 1: 클래스 간 차이를 가장 잘 구분하는 주축 (주로 setosa vs 나머지)
	# LDA Component 2: 두 번째로 중요한 구분 축 (versicolor vs virginica 구분 보조)
	# 원래 4차원 데이터(꽃받침·꽃잎 길이/너비)가 2차원 선형결합(축소공간)으로 투영되었음에도 세 품종의 경계가 명확히 드러남
	# --------------------------------------------------
	# (추가 참고)
	# print("Explained variance ratio:", lda.explained_variance_ratio_)
	#   → 각 LDA 성분이 클래스 분리를 설명하는 비율
	# print("Scalings:", lda.scalings_)
	#   → 각 원변수(4개)가 LDA 축에 기여하는 가중치(선형결합 계수)


**([1-1] LDA 예제 소스 실행결과)**

![](./images/LDA.png)
<br>


![](./images/PCA_LDA.png)
<br>
https://nirpyresearch.com/classification-nir-spectra-linear-discriminant-analysis-python/


**(PCA vs LDA 예제 소스)**

	"""
	Bream(도미) vs Smelt(빙어) 실데이터(원격 로드 실패 시 내장 표본)로 PCA vs LDA 비교
	- 사용 특성: Length2(중간길이), Height
	"""
	
	import io
	import textwrap
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	
	# -------------------- 1) 원격 로드 + 오프라인 폴백 --------------------
	URLS = [
	    "https://raw.githubusercontent.com/selva86/datasets/master/Fish.csv",
	    "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/Fish.csv",  # 예비
	]
	
	# 공개 Fish Market에서 Bream/Smelt 행 일부만 발췌(Length2, Height 중심)
	FALLBACK_CSV = textwrap.dedent("""\
	Species,Length2,Height
	Bream,23.2,11.52
	Bream,24.0,12.48
	Bream,23.9,12.37
	Bream,26.3,12.73
	Bream,26.5,14.18
	Bream,29.0,14.73
	Bream,29.7,14.88
	Bream,29.9,17.78
	Bream,31.0,16.24
	Bream,31.5,16.64
	Bream,32.0,15.05
	Bream,33.0,15.58
	Bream,33.5,18.26
	Bream,35.0,18.49
	Bream,36.5,18.18
	Bream,36.0,18.67
	Bream,39.0,19.99
	Bream,41.0,21.06
	Smelt,12.9,3.52
	Smelt,14.5,3.52
	Smelt,13.2,4.30
	Smelt,14.3,4.23
	Smelt,15.0,5.14
	Smelt,16.2,5.58
	Smelt,17.4,5.52
	Smelt,17.4,5.22
	Smelt,19.0,5.20
	Smelt,19.0,5.58
	Smelt,20.0,5.69
	Smelt,20.5,5.92
	Smelt,21.0,6.11
	Smelt,22.0,6.63
	""")
	
	def load_fish_df():
	    last_err = None
	    for url in URLS:
	        try:
	            df = pd.read_csv(url)
	            # 셀바86 데이터셋 스키마 확인
	            if {"Species","Length2","Height"}.issubset(df.columns):
	                return df
	        except Exception as e:
	            last_err = e
	            continue
	    # 폴백: 내장 표본 사용
	    df = pd.read_csv(io.StringIO(FALLBACK_CSV))
	    return df
	
	df = load_fish_df()
	df = df[df["Species"].isin(["Bream","Smelt"])].copy()
	
	# -------------------- 2) 특징 선택/표준화 --------------------
	features = ["Length2","Height"]
	X = df[features].to_numpy().astype(float)
	y = (df["Species"]=="Bream").astype(int).to_numpy()  # Bream=1, Smelt=0
	
	scaler = StandardScaler()
	Xz = scaler.fit_transform(X)
	
	# -------------------- 3) PCA/LDA 축 계산 --------------------
	pca = PCA(n_components=1).fit(Xz)
	w_pca = pca.components_[0]  # (2,)
	
	lda = LDA(n_components=1).fit(Xz, y)
	w_lda = lda.scalings_.ravel()
	w_lda = w_lda / np.linalg.norm(w_lda)
	
	def project_perp(P, w):
	    w = w / np.linalg.norm(w)
	    t = P @ w
	    return np.outer(t, w)
	
	def endpoints(w, span=5.5):
	    w = w/np.linalg.norm(w)
	    return np.vstack([-span*w, span*w])
	
	def plot_panel(ax, X, y, w, title, subtitle):
	    P = X
	    Pr = project_perp(P, w)
	
	    ax.scatter(P[y==1,0], P[y==1,1], c="crimson", s=36, label="Bream")
	    ax.scatter(P[y==0,0], P[y==0,1], c="royalblue", s=36, label="Smelt")
	
	    ab = endpoints(w, 5.5)
	    ax.plot(ab[:,0], ab[:,1], "k-", lw=2)
	    ax.arrow(0,0, w[0]*2.2, w[1]*2.2, head_width=0.15, head_length=0.22, fc="k", ec="k")
	
	    # 수직투영(회색 점선)
	    for p, q in zip(P, Pr):
	        ax.plot([p[0], q[0]], [p[1], q[1]], ls="--", c="gray", lw=1, alpha=0.9)
	
	    ax.scatter(Pr[y==1,0], Pr[y==1,1], c="crimson", s=16)
	    ax.scatter(Pr[y==0,0], Pr[y==0,1], c="royalblue", s=16)
	
	    ax.set_aspect("equal","box")
	    ax.set_xlabel(f"{features[0]} (z-score)")
	    ax.set_ylabel(f"{features[1]} (z-score)")
	    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
	    ax.grid(False)
	    ax.legend(fontsize=9, loc="upper left")
	    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
	
	# -------------------- 4) 플롯 --------------------
	fig, axes = plt.subplots(1,2, figsize=(10,6))
	plot_panel(axes[0], Xz, y, w_pca,
	           "PCA projection:", "Maximising the variance of the whole set")
	plot_panel(axes[1], Xz, y, w_lda,
	           "LDA projection:", "Maximising the distance between groups")
	plt.tight_layout(); plt.show()
	
	# -------------------- 5) 분리도 수치 비교 --------------------
	def fisher_score_1d(z, y):
	    m1, m0 = z[y==1].mean(), z[y==0].mean()
	    s1, s0 = z[y==1].var(ddof=1), z[y==0].var(ddof=1)
	    return (m1 - m0)**2 / (s1 + s0)
	
	z_pca = Xz @ (w_pca/np.linalg.norm(w_pca))
	z_lda = Xz @ (w_lda/np.linalg.norm(w_lda))
	print("[Fisher 분리 점수] (값 ↑ = 분리 ↑)")
	print(f"PCA 축 : {fisher_score_1d(z_pca, y):.3f}")
	print(f"LDA 축 : {fisher_score_1d(z_lda, y):.3f}")


**(PCA vs LDA 예제 소스 실행 결과)**

	[Fisher 분리 점수] (값 ↑ = 분리 ↑)
	PCA 축 : 8.604
	LDA 축 : 21.683

![](./images/PCA_vs_LDA.png)


**(PCA vs LDA 예제 소스 실행 결과 분석)**

	[Fisher 분리점수] (값 ↑ = 분리 ↑) : 두 클래스의 중심이 멀리 떨어져 있고, 각 클래스 내부의 분산이 작을수록 값이 커진다.(전처리에 사용)
	PCA 축 : 8.604    → (비지도) PCA는 분산 최대가 되는 축 : 전체분산 최대화
	                     빨강/파랑을 구분하지 않고, 가장 퍼져 보이는 방향(화살표)을 찾은 뒤 그 축 위로 직선 투영
	LDA 축 : 21.683   → (지도) LDA는 집단 분리가 최대가 되는 축 : 집단간분산/집단내분산 최대화
	                     각 집단이 축 위에서 멀어지도록 하면서, 같은 집단 내부는 모이도록(퍼짐 최소) 하는 방향 찾기

<br>

# [1-2] Quadratic Discriminant Analysis (QDA)
▣ 정의 : 새로운 데이터가 어느 클래스(집단)에 속할지 예측하는 분류 알고리즘으로<br>
데이터가 여러 그룹으로 나뉘어 있을 때, 각 그룹의 확률 분포(특히 평균과 공분산)를 이용해서 “이 점은 어떤 그룹에서 나올 가능성이 가장 높을까?”를 계산하는 방식<br>
(가정 : 각 클래스의 공분산이 다를 수 있다. 모양이 다른 2차식 곡선)<br>
▣ 목적 : 클래스 간의 구조가 더 복잡하고 선형 경계로는 충분히 분리되지 않을 때, 좀 더 유연한 분리 경계를 제공<br>
▣ 장점 : 공분산이 클래스마다 다를 경우 LDA보다 유연하게 분류 성능이 향상, 비선형(곡선) 경계도 허용하므로 복잡한 데이터 구조에 대응 가능<br>
▣ 단점 : 클래스별 공분산을 추정해야 하므로 샘플 수가 충분치 않거나 고차원 특성일 경우 과적합 및 수치불안정 문제가 발생<br>
▣ Scikit-learn 클래스명 : sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis<br> 
▣ 가이드 : https://scikit-learn.org/stable/modules/lda_qda.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html<br>

**(LDA와 QDA 응용분야 비교)**
| 구분            | **LDA (선형판별분석)**                                                                                                                 | **QDA (이차판별분석)**                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **적용 데이터 형태** | 각 클래스가 비슷한 분포(타원 모양), 경계가 직선으로 나뉨                                                                                                | 클래스마다 분포 모양이 다르고, 경계가 곡선형                                                                                                             |
| **데이터 특성**    | 단순하고, 클래스 간 경계가 ‘직선적’                                                                                                            | 복잡하고, 경계가 ‘비선형적’                                                                                                                      |
| **필요한 데이터량**  | 적은 데이터에도 안정적 (공분산을 하나만 추정)                                                                                                       | 데이터가 많아야 함 (클래스별 공분산을 따로 추정)                                                                                                          |
| **계산 복잡도**    | 낮음 (모수 적음)                                                                                                                       | 높음 (모수 많음)                                                                                                                            |
| **대표 응용 분야**  | • 얼굴인식 (초기형 Face Recognition)  <br>• 텍스트 분류 (Spam vs Ham) <br>• 의료 데이터 진단 (정상/비정상 구분) <br>• 품질관리, 결함탐지 <br>• 마케팅 고객 세분화(단순군집 기반) | • 음성 인식 (성별, 감정 분류 등) <br>• 생물정보학(유전자 발현 데이터) <br>• 복잡한 이미지 분류(비선형 경계) <br>• 비정상 탐지(Anomaly Detection) <br>• 금융 리스크 예측 (클래스 분산이 다를 때) |
| **모델 형태**     | 선형 경계 (직선·평면)                                                                                                                    | 곡선형 경계 (포물선·타원형)                                                                                                                      |
| **적합한 상황**    | 변수 간 관계가 선형, 공분산 구조가 유사                                                                                                          | 클래스 간 공분산 구조가 다름, 비선형 구조                                                                                                              |
| **예시 데이터**    | Iris 데이터(두 클래스 구분 선형 가능)                                                                                                         | 복잡한 패턴의 음성·영상 데이터                                                                                                                     |
| **장점**        | 빠르고 단순, 해석 용이                                                                                                                    | 유연하고 복잡한 경계 표현 가능                                                                                                                     |
| **단점**        | 비선형 데이터에 부적합                                                                                                                     | 과적합 위험, 계산량 큼                                                                                                                         |

<br>

# [2-1] Principal Component Regression (PCR)
<br>
▣ 정의 : 먼저 독립변수 𝑋에 대해 Principal Component Analysis(PCA)를 적용하여 비지도학습의 차원축소(주성분)를 수행하고,<br> 
그 다음 주성분을 독립변수로 하여 선형회귀(OLS 등)를 수행하는 이중 단계 방식의 회귀기법<br> 
▣ 목적 : 다중공선성(multicollinearity) 문제가 크거나, 변수차원이 매우 큰 경우에 차원을 축소함으로써 회귀 안정성을 확보하고 과적합을 완화<br> 
▣ 장점 : 공선성이 심한 데이터나 변수수가 매우 많은 상황에서 유용, 차원축소→회귀 단계를 통해 모델 단순화 및 해석 가능성 제고<br>
▣ 단점 : 주성분 선택 시 ‘변동성(variance)’ 큰 주성분이 반드시 예측력(종속변수 설명력)이 높은 것은 아니라는 점에서, 중요한 정보가 사라질 가능성<br> 
비지도 방식의 PCA를 먼저 수행하므로, 종속변수 𝑦정보가 주성분 선정에 반영되지 않아 예측력이 떨어질 가능성<br>
▣ Scikit-learn 클래스명 : (파이프라인) sklearn.decomposition.PCA + sklearn.linear_model.LinearRegression<br> 
▣ 가이드 : https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html<br>
 
<br>

# [2-2] Partial Least Squares (PLS)

▣ 정의 : 독립변수 𝑋와 종속변수 y 양쪽을 고려하여 새로운 잠재변수(성분)를 추출하고, 이 잠재변수를 기반으로 회귀모형을 적합하는 차원축소 회귀기법<br> 
▣ 목적 : 독립변수 수가 많고 다중공선성이 심하거나, 관측치 수 < 변수 수인 고차원 상황에서 𝑋와 y 간의 공변량 구조를 최대한 반영하면서 회귀모형을 구축<br> 
▣ 장점 : 𝑋와 y 간의 상관/공변량을 고려하므로, PCR보다 종속변수 설명력을 더 잘 확보, 차원축소와 회귀를 동시에 수행하여 고차원/공선성 데이터에서 안정적<br>
▣ 단점 : 해석이 다소 복잡하고, 잠재변수 구성 방식이 덜 직관적일 가능성, 구성 성분 수(n_components)가 과다하게 선택하면 과적합 위험도 존재<br>
▣ Scikit-learn 클래스명 : sklearn.cross_decomposition.PLSRegression<br> 
▣ 가이드 : https://scikit-learn.org/stable/modules/cross_decomposition.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html<br>
 
<br>

# [2-3] PLS‑DA (Partial Least Squares Discriminant Analysis)
▣ 정의 : PLS 기법을 변형하여 **종속변수가 범주형(y가 클래스 레이블)**인 경우에 적용하는 판별분석 형태의 기법<br> 
▣ 목적 : PLS의 잠재변수 추출 방식과 판별분석 배치를 결합해, 고차원/공선성 있는 데이터에서 분류모델을 구축<br>
▣ 장점 : 전통적인 판별모델(LDA/QDA)보다 변수 수가 많거나 특성 간 상관이 높을 때 유리<br>
▣ 단점 : scikit-learn에서 공식적으로 독립된 “PLS-DA” 클래스가 제공되지 않음. 잠재변수 해석이 어렵고, 튜닝이 복잡할 가능성<br>
▣ Scikit-learn 클래스명 : 공식 클래스 없음 → 일반적으로 PLSRegression + 범주형 y → 후처리 판별분석 형태로 구현<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/cross_decomposition.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html<br>

<br>

# [2-4] Supervised PCA
▣ 정의 : 일반 PCA가 독립변수 𝑋만을 고려해 주성분을 추출하는 데 반해, 종속변수 𝑦 정보까지 이용해 차원축소를 수행하는 방식(즉, 지도형 차원축소)<br>
▣ 목적 : 차원축소하면서도 𝑦와의 관계(예측력)를 보존하려는 목적<br>
▣ 장점 : 단순 PCA보다 예측모델 성능을 향상, 변수 수가 많고 예측변수→종속변수 간 관계가 복잡할 때 유리<br>
▣ 단점 : scikit-learn에서 하나의 표준 클래스명으로 제공되지는 않아 구현에 유연성이 필요, 해석이 다소 어렵고, 과적합 가능성<br>
▣ Scikit-learn 클래스명 : 공식 제공 없음<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/decomposition.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html<br>
 
<br>

# [3-1] 결정 트리 (Decision Tree)
▣ 정의 : 독립변수 공간을 반복적으로 분할(split)하여 리프 노드(leaf)에서 예측값을 출력하는 트리구조의 지도학습 모델<br> 
▣ 목적 : 입력 변수의 분할 기준을 찾아 복잡한 비선형 관계를 모델링하고, 직관적인 규칙 기반 예측모델을 제공<br>
▣ 장점 : 해석이 쉽고 트리 시각화 등을 통해 설명 가능, 변수 변환이나 스케일링이 크게 필요 없으며, 비선형 관계나 변수 상호작용을 자연스럽게 반영<br>
▣ 단점 : 과적합 위험이 크고, 세세하게 튜닝하지 않으면 일반화 성능 저하 가능성, 트리가 너무 깊거나 분할 기준이 복잡해지면 해석이 어려워질 가능성<br>
▣ Scikit-learn 클래스명 : 분류용 sklearn.tree.DecisionTreeClassifier, 회귀용 sklearn.tree.DecisionTreeRegressor<br> 
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html<br>
▣ API : https://scikit-learn.org/stable/auto_examples/tree/index.html<br>

![](./images/tree.png)

| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 시각화를 통한 해석의 용이성(나무 구조로 표현되어 이해가 쉬움, 새로운 개체 분류를 위해 루트 노드부터 끝 노트까지 따라가면 되므로 분석 용이) | 휴리스틱에 근거한 실용적 알고리즘으로 학습용 자료에 의존하기에 전역 최적화를 얻지 못할 수도 있음(검증용 데이터를 활용한 교차 타당성 평가를 진행하는 과정이 필요) |
| 데이터 전처리, 가공작업이 불필요 | 자료에 따라 불안정함(적은 수의 자료나 클래스 수에 비교하여 학습 데이터가 적으면 높은 분류에러 발생) | 
| 수치형, 범주형 데이터 모두 적용 가능 | 각 변수의 고유한 영향력을 해석하기 어려움 | 
| 비모수적인 방법으로 선형성, 정규성 등의 가정이 필요없고 이상값에 민감하지 않음 | 자료가 복잡하면 실행시간이 급격하게 증가함 | 
| 대량의 데이터 처리에도 적합하고 모형 분류 정확도가 높음 | 연속형 변수를 비연속적 값으로 취급하여 분리 경계점에서는 예측오류가 매우 커지는 현상 발생 | 

![](./images/trees.png)

<br>

## 결정 트리 회귀(Decision Tree Regression)
▣ 정의 : 데이터에 내재되어 있는 패턴을 비슷한 수치의 관측치 변수의 조합으로 예측 모델을 나무 형태로 만든다.<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html<br>
▣ 모델식 : $\widehat{f}(x) = \sum_{m=1}^{M}C_mI((x_1,x_2)\in R_m)$<br>
비용함수(cost function)를 최소로 할때 최상의 분할 : 데이터를 M개로 분할($R_1,R_2,...R_M$)<br> 
$\underset{C_m}{min}\sum_{i=1}^{N}(y_i-f(x_i))^2=\underset{C_m}{min}\sum_{i=1}^{N}(y_i-\sum_{m=1}^{M}C_mI(x\in R_m))^2$<br>
각 분할에 속해 있는 y값들의 평균으로 예측했을때 오류가 최소화 : $\widehat{C}_m=ave(y_i|x_i\in R_m)$<br>

**(Decision Tree Regression 예제 소스)**

	# ============================================
	# 결정트리 회귀(DecisionTreeRegressor) 예제 (완전 실행형)
	# 데이터: sklearn 내장 당뇨(회귀용) 데이터셋
	# 절차: 데이터 로드 → 학습/테스트 분할 → 모델 학습 → 예측 → 평가
	# 핵심 포인트: 트리계열은 스케일링이 필수는 아님(분할 기준이 순위/임계값 기반) max_depth 등 하이퍼파라미터로 과적합 제어
	# ============================================
	from sklearn.datasets import load_diabetes
	from sklearn.model_selection import train_test_split
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
	import numpy as np

	# --------------------------------------------------
	# 1) 데이터 로드 (회귀용: 연속형 타깃 y)
	# --------------------------------------------------
	diabetes = load_diabetes()
	X = diabetes.data        # shape (442, 10) — 10개의 수치형 특징
	y = diabetes.target      # shape (442,)     — 질병 진행 정도(연속값)

	# --------------------------------------------------
	# 2) 학습/테스트 분할
	#    - random_state 고정: 재현성 보장
	#    - test_size=0.2: 20%를 테스트로 사용
	# --------------------------------------------------
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

	# --------------------------------------------------
	# 3) 모델 생성 및 학습
	#    - max_depth=5: 트리 최대 깊이 제한(과적합 방지용)
	#    - random_state=42: 분할/동작의 재현성
	# --------------------------------------------------
	tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
	tree_reg.fit(X_train, y_train)

	# --------------------------------------------------
	# 4) 예측
	# --------------------------------------------------
	y_pred = tree_reg.predict(X_test)

	# --------------------------------------------------
	# 5) 성능 평가
	#    - MSE: 평균 제곱 오차 (작을수록 좋음)
	#    - RMSE: 제곱근(해석 편의, y 단위와 동일)
	#    - MAE: 평균 절대 오차 (이상치에 덜 민감)
	#    - R2 : 결정계수 (1에 가까울수록 설명력 높음)
	# --------------------------------------------------
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error (MSE): {mse:.3f}")
	print(f"Root MSE (RMSE)        : {rmse:.3f}")
	print(f"Mean Absolute Error    : {mae:.3f}")
	print(f"R^2 Score              : {r2:.3f}")

	# --------------------------------------------------
	# (선택) 특징 중요도 확인: 어떤 변수로 분할을 많이 했는지
	# --------------------------------------------------
	importances = tree_reg.feature_importances_
	# 중요도가 0이 아닌 상위 특징만 보기
	top_idx = np.argsort(importances)[::-1]
	print("\n[Feature Importances]")
	for i in top_idx:
    	if importances[i] > 0:
	        print(f"- {diabetes.feature_names[i]:>6s}: {importances[i]:.3f}")

	# ============================================
	# [해설]
	# - NameError 해결: train_test_split로 X_train, X_test, y_train, y_test를 먼저 생성해야 함.
	# - 트리의 일반화 성능은 max_depth, min_samples_leaf, min_samples_split 등을 조절하며 개선 가능.
	# - 트리는 스케일링 영향이 작지만, 데이터가 매우 잡음이 크거나 샘플이 적으면 과적합되기 쉬움.
	#   필요 시 max_depth를 더 줄이거나 min_samples_leaf를 늘려 규제하세요.
	# ============================================


**(Decision Tree Regression 예제 소스 실행 결과)**

	Mean Squared Error (MSE): 3526.016
	Root MSE (RMSE)        : 59.380
	Mean Absolute Error    : 45.937
	R^2 Score              : 0.334

	[Feature Importances]
	-    bmi: 0.555
	-     s5: 0.189
	-     s1: 0.062
	-     s6: 0.059
	-     s4: 0.040
	-    age: 0.032
	-     s3: 0.023
	-     bp: 0.022
	-     s2: 0.017
	-    sex: 0.002


**(Decision Tree Regression 예제 소스 실행 결과 분석)**

	Mean Squared Error(MSE) : 3526.016 → 예측값과 실제값의 평균 제곱 오차로 모델 오차가 다소 큰 편
	Root MSE(RMSE)          : 59.380 → 예측 오차의 평균 크기가 약 ±59 단위 정도(당뇨 진행 지수는 0~300 정도이므로 오차가 중간 수준)
	Mean Absolute Error(MAE): 45.937 → 평균적으로 약 45.9 정도 차이(MAE < 30 우수, 30 ≤ MAE ≤ 50 중간, MAE > 70 부정확)
	R^2 Score(결정계수)       : 0.334 → 전체 분산의 약 33.4%만 설명(예측력이 제한적) 과적합없이 기본트리 모델로는 중간수준의 성능

	[Feature Importances]
	-    bmi: 0.555  → 가장 높은 중요도를 가짐. 비만도가 높을수록 인슐린 저항성과 혈당 수치가 증가하므로 당뇨 진행 정도 예측에 절대적 영향을 미침. 트리의 루트 분할(첫 기준)로 사용되었을 가능성
	-     s5: 0.189  → 혈중 지질(특히 중성지방) 대사를 나타내며, 지질대사 이상과 인슐린 저항성 간의 연관성 반영. 혈지질 수치가 높을수록 당뇨 악화 위험 증가
	-     s1: 0.062  → 총콜레스테롤 수치로, 고지혈증·혈관계 문제와 관련. 혈중 콜레스테롤이 높을수록 당뇨 합병증 위험 상승
	-     s6: 0.059  → 혈당과 직접 관련된 변수. 실제 혈당 농도 변화가 당뇨 진행에 직접적으로 반영됨. s5와 함께 대사성 특징을 설명
	-     s4: 0.040  → 혈청 인슐린 반응성을 나타내며, 인슐린 분비 기능 저하 여부를 반영. 당대사 불균형이 심한 환자에서 값이 크게 작용
	-    age: 0.032  → 고령일수록 당뇨 발생 및 진행 위험이 커짐. 다만 다른 생리적 요인(BMI, 지질 수치 등)에 비해 직접적인 영향은 상대적으로 작음
	-     s3: 0.023  → 좋은 콜레스테롤로 낮을수록 심혈관 질환 및 당뇨 합병증 위험이 증가. 모델에서는 보조적 지표로 사용
	-     bp: 0.022  → 혈압은 인슐린 저항성과 연관. 고혈압·대사증후군 환자에서 당뇨 진행 속도 가속화. 모델에서 보조 설명변수로 작용
	-     s2: 0.017  → 나쁜 콜레스테롤로 높을수록 혈관손상 및 당뇨합병증 유발 가능성 높음. 영향도는 크지 않지만 지질대사 요인의 일부로 반영
	-    sex: 0.002  → 남녀 간 평균적 대사차는 있지만, 이 데이터셋에서는 큰 영향 없음. 모델에서 거의 사용되지 않음
	
<br>

## 결정 트리 분류(Decision Tree Classification)
▣ 정의 : 데이터에 내재되어 있는 패턴을 비슷한 범주의 관측치 변수의 조합으로 분류 모델을 나무 형태로 만든다.<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/tree.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html<br>
▣ 모델식 : $\widehat{f(x)} = \sum_{m=1}^{M}k(m)I((x_1,x_2)\in R_m)$<br>
끝노드(m)에서 클래스(k)에 속할 관측치의 비율 : $\widehat{P_{mk}}=\frac{1}{N_m}\sum_{x_i\in R_m}^{}I(y_i=k)$<br>
끝노드 m으로 분류된 관측치 : $k(m) = \underset{k}{argmax}\widehat{P_{mk}}$<br><br>
▣ 비용함수(불순도 측정) : 불순도(Impurity)가 높을수록 다양한 클래스들이 섞여 있고, 불순도가 낮을수록 특정 클래스에 속한 데이터가 명확<br>
(1) 오분류율(Misclassification rate, Error rate) : 분류 모델이 잘못 분류한 샘플의 비율로, 전체 샘플 중에서 실제 값과 예측 값이 일치하지 않는 샘플의 비율을 나타낸다.(0에서 100 사이의 값, 0%: 모델이 모든 샘플을 완벽하게 예측, 100%: 모델이 모든 샘플을 잘못 예측)<br><br>
$\frac{FP+FN}{TP+TN+FP+FN}$ 
###### FP(False Positive) : 실제 값이 Negative인데 Positive로 예측, FN(False Negative) : 실제 값이 Positive인데 Negative로 예측, TP(True Positive) : 실제 값이 Positive이고 Positive로 올바르게 예측, TN(True Negative) : 실제 값이 Negative이고 Negative로 올바르게 예측<br>
(2) 지니계수(Gini Coefficient) : 데이터셋이 얼마나 혼합되어 있는지를 나타내는 불순도의 측정치(0에서 0.5 사이의 값, 0: 데이터가 완벽하게 한 클래스에 속해 있음을 의미하며, 불순도가 전혀 없는 상태, 0.5: 두 개의 클래스가 완벽하게 섞여 있는 상태)<br>
$Gini(p)=1-\sum_{i=1}^{n}p_i^2$<br><br>
(3) 엔트로피(Entropy) : 확률 이론에서 온 개념으로 불확실성 또는 정보의 무질서를 측정하는 또 다른 방식으로, 데이터가 얼마나 혼란스럽고 예측하기 어려운지를 측정(0에서 1 사이의 값, 0 : 데이터가 완벽하게 한 클래스에 속해 있으며, 불확실성이 없는 상태, 1 : 데이터가 완전히 섞여 있고, 가장 큰 불확실성을 가지고 있다.<br>
$Entropy(p) = -\sum_{i=1}^{n}p_ilog_2p_i$<br> 

▣ 유형 :  ID3, CART
 - ID3 : 모든 독립변수가 범주형 데이터인 경우에만 분류가 가능하다. 정보획득량(Infomation Gain)이 높은 특징부터 분기해나가는데 정보획득량은 분기전 엔트로피와 분기후 엔트로피의 차이를 말한다.(엔트로피 사용)<br><br>
$IG(S, A) = E(S) - E(S|A)$<br>
 - CART : Classification and Regression Tree의 약자로, 이름 그대로 분류와 회귀가 모두 가능한 결정트리 알고리즘으로 yes 또는 no 두 가지로 분기한다.(지니계수 사용)<br><br> 
$f(k,t_k) = \frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}$<br>

<br>

	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 	#결정 트리 분류 모델 생성 (최대 깊이 3으로 설정)
	clf = DecisionTreeClassifier(max_depth=3, random_state=42)
	clf.fit(X_train, y_train)  # 학습 데이터를 이용해 모델 학습

	#테스트 데이터에 대한 예측
	y_pred = clf.predict(X_test)

	#정확도 출력
	accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
	print(f"Accuracy: {accuracy * 100:.2f}%")  # 정확도 출력


<br>

	(개별 트리 모델의 단점)	
 	계층적 구조로 인해 중간에 에러가 발생하면 다음 단계로 에러가 계속 전파
  	학습 데이터의 미세한 변동에도 최종결과에 큰 영향
   	적은 개수의 노이즈에도 큰 영향
	나무의 최종 노드 개수를 늘리면 과적합 위함(Low Bias, Large Variance)

	(해결방안) 랜덤 포레스트(Random forest)


<br>

# [3-2] 랜덤 포레스트 (Random Forest)
▣ 정의 : 많은 트리를 무작위로 만들어 다수결로 예측하는 방법<br>
여러 개의 Decision Tree를 배깅(Bagging, Bootstrap Aggregating) 방식으로 학습하여,<br>
그 예측값을 평균(회귀) 또는 다수결(분류)로 통합하는 앙상블(Ensemble) 학습 알고리즘<br>
각 트리는 서로 다른 부트스트랩 표본과 일부 특성(feature subset)을 사용하여 모델 간 상관을 줄이고, 과적합(overfitting)을 완화<br>
▣ 목적 : 단일 결정 트리의 불안정성(variance 높음)을 보완하고, 예측의 안정성(stability)과 정확도(accuracy)를 높이기<br>
▣ 장점 : 트리 여러 개를 평균/투표함으로써 분산(variance)을 낮춰서 과적합 방지, 변수 중요도(Feature Importance) 자동 산출, 비선형 관계 및 변수 간 상호작용을 자연스럽게 포착, 데이터 스케일 조정 불필요, 결측값에도 비교적 강건, 분류와 회귀 모두 사용 가능하며, 이상치(outlier)에 민감하지 않음<br>
▣ 단점 : 개별 트리 수가 많아 모델 해석이 어렵고, 많은 트리 수로 훈련과 예측시간이 길어짐(메모리 및 연산량 증가), 트리 간 상관성 완전 제거 불가<br>
▣ Scikit-learn 클래스명 : 분류용 sklearn.ensemble.RandomForestClassifier 회귀용 sklearn.ensemble.RandomForestRegressor<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/ensemble.html#random-forests<br>
▣ 모델식 : $\widehat{y}=\frac{1}{N}\sum_{i=1}^{N}T_i(X)$ ($N$ : 결정트리의 수, $T_i(X)$ : 각 결정트리 $i$가 입력값 $X$에 대해 예측한 값)

![](./images/Bootstrap.png)
<br>
출처: https://www.researchgate.net/figure/Schematic-of-the-RF-algorithm-based-on-the-Bagging-Bootstrap-Aggregating-method_fig1_309031320<br>


| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 모델이 단순, 과적합이 잘 일어나지 않음 | 여러개의 결정트리 사용으로 메모리 사용량 큼 |
| 새로운 데이터에 일반화가 용이함 | 고차원 및 희소 데이터에 잘 작동하지 않음 |

## 랜덤 포레스트 회귀(Random Forest Regression)  
▣ 정의 : 각 트리가 예측한 값들의 평균을 통해 최종 예측값을 도출하는 모델로, 다수결 대신 트리에서 얻은 예측값의 평균을 사용하여 연속값 예측<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#randomforestregressor<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor<br>
▣ 모델식 : $\widehat{y}= \frac{1}{B}\sum_{i=1}^{B}T_i(x)$<br>
###### $T_i(x)$: 입력 데이터 𝑥에 대한 𝑖번째 결정 트리의 예측값, B: 전체 트리의 개수


	from sklearn.ensemble import RandomForestRegressor
 
 	# 트리 개수를 변화시키며 모델 학습 및 평가
	for iTrees in nTreeList:
    		depth = None  # 트리 깊이 제한 없음
    		maxFeat = 4  # 사용할 최대 특징 수
    		# 랜덤 포레스트 회귀 모델 생성 및 학습
    		wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
			max_depth=depth, max_features=maxFeat,
			oob_score=False, random_state=531)
    		wineRFModel.fit(xTrain, yTrain)  # 모델 학습
    		# 테스트 데이터에 대한 예측값 계산
    		prediction = wineRFModel.predict(xTest)
    		# MSE 계산 및 누적
    		mseOos.append(mean_squared_error(yTest, prediction))
     
	# MSE 출력
	print("MSE")
	print(mseOos)


<br>

## 랜덤 포레스트 분류(Random Forest Classification)    	  	
▣ 정의 : 다수의 Decision Trees를 기반으로 한 앙상블 모델로, 각 나무는 독립적으로 클래스를 예측한 후 다수결 투표를 통해 최종 클래스를 결정<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#randomforestclassifier<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier<br>
▣ 모델식 : $\widehat{y}=mode(T_1(x),T_2(x),...,T_B(x))$<br>
###### $T_i(x)$: 입력 데이터 𝑥에 대한 𝑖번째 결정 트리의 예측값, B: 전체 트리의 개수, mode 함수 : 다수결 투표방식

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn import datasets

	# 붓꽃 데이터셋 로드
	iris = datasets.load_iris()

	# 독립 변수와 종속 변수 분리
	X = iris.data
	y = iris.target

	# 학습용 데이터와 테스트용 데이터 분리
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 랜덤 포레스트 모델 초기화
	model = RandomForestClassifier()

	# 모델 학습
	model.fit(X_train, y_train)

	# 테스트 데이터 예측
	y_pred = model.predict(X_test)

	# 정확도 계산
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)
 

<br>

# [4-1] k-최근접 이웃(k-Nearest Neighbors, K-NN)
▣ 정의 : 머신러닝에서 데이터를 가장 가까운 유사속성에 따라 분류하여 데이터를 거리기반으로 분류분석하는 기법으로,<br>
비지도학습인 군집화(Clustering)과 유사한 개념이나 기존 관측치의 y 값이 존재한다는 점에서 지도학습에 해당<br>
새로운 입력 샘플에 대해 학습데이터 중 가장 가까운 𝑘개의 이웃을 찾아, 이들의 레이블(분류)이나 평균(회귀)을 이용해 예측하는 비모수 기반의 지도학습 모델<br> 
▣ 목적 : 단순하면서도 학습된 모델 구조가 거의 없으므로 빠르게 적용 가능하고, 데이터의 형태가 복잡하거나 비선형일 때 유연하게 대응하고자 할 때 사용<br>
▣ 장점 : 학습 단계가 거의 없고, 구현이 매우 간단, 비선형 경계나 복잡한 데이터 구조를 자연스럽게 모델링 가능<br>
▣ 단점 : test 비용이 상대적으로 크며, 고차원 특성공간에서는 거리 측정왜곡(차원의 저주)으로 성능저하, 적절한 𝑘와 거리 메트릭 선택이 중요하며, 이상치나 노이즈 민감성<br>
▣ Scikit-learn 클래스명 : 분류용 sklearn.neighbors.KNeighborsClassifier 회귀용 sklearn.neighbors.KNeighborsRegressor<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/neighbors.html<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html<br>


| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 간단하고 이해하기 쉬움  | 모델 미생성으로 특징과 클래스 간 관계 이해가 제한적 |
| 학습 데이터분포 고려 불요 | 적절한 K의 선택이 필요 |
| 빠른 훈련 단계 | 데이터가 많아지면 느림 : 차원의 저주(curse of dimensionality) |
| 수치기반 데이터 분류 성능우수 | 명목특징 및 누락데이터위한 추가처리 필요(이상치에 민감)|

데이터로부터 거리가 가까운 'K'개의 다른 데이터의 레이블을 참조하여 분류할때 거리측정은 유클리디안 거리 계산법을 사용<br>
![](./images/distance.PNG)

K-NN 모델은 각 변수들의 범위를 재조정(표준화, 정규화)하여 거리함수의 영향을 줄여야 한다.<br>
(1) 최소-최대 정규화(min-max normalization) : 변수 X의 범위를 0(0%)에서 1(100%)사이로 나타냄<br><br>
$X_{new} = \frac{X-min(X)}{max(X)-min(X)}$<br>

(2) z-점수 표준화(z-score standardization) : 변수 X의 범위를 평균의 위또는 아래로 표준편차만큼 떨어져 있는 지점으로 확대 또는 축소(데이터를 평균 0, 표준편차 1로 변환)하는 방식으로, 데이터의 중심을 0으로 맞추고, 데이터를 단위 표준 편차로 나누어 값을 재조정<br><br>
$X_{new} = \frac{X-\mu}{\sigma}= \frac{X-min(X)}{StdDev(X)}$

<br>

## k-최근접 이웃 회귀(k-Nearest Neighbors Regression)
▣ 정의 :주변의 가장 가까운 K개의 샘플 평균을 통해 값을 예측하는 방식<br> 
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html<br>
▣ 한계 : 테스트하고자 하는 샘플에 근접한 훈련 데이터가 없는 경우, 즉 훈련 셋의 범위를 많이 벗어나는 샘플인 경우 정확하게 예측하기 곤란<br> 

	class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', 
	leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
	 
	# n_neighbors : int
	# 이웃의 수인 K를 결정한다. default는 5다. 
	 
  	# weights : {'uniform', 'distance'} or callable
	# 예측에 사용되는 가중 방법을 결정한다. default는 uniform이다. 
	# 'uniform' : 각각의 이웃이 모두 동일한 가중치를 갖는다. 
	# 'distance' : 거리가 가까울수록 더 높은 가중치를 가져 더 큰 영향을 미치게 된다.
	# callable : 사용자가 정의한 함수(거리가 저장된 배열을 입력받고 가중치가 저장된 배열을 반환)
 	
	# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'} 
	# 가장 가까운 이웃들을 계산하는 데 사용하는 알고리즘을 결정한다. default는 auto이다. 
	# 'auto' : 입력된 훈련 데이터에 기반하여 가장 적절한 알고리즘을 사용한다. 
	# 'ball_tree' : Ball-Tree 구조를 사용한다.(https://nobilitycat.tistory.com/entry/ball-tree)
	# 'kd_tree' : KD-Tree 구조를 사용한다.
	# 'brute' : Brute-Force 탐색을 사용한다.  	
 	
	# leaf_size : int
	# Ball-Tree나 KD-Tree의 leaf size를 결정한다. default값은 30이다.
	# 트리를 저장하기 위한 메모리뿐만 아니라, 트리의 구성과 쿼리 처리의 속도에도 영향을 미친다. 
 	
	# p : int
	# 민코프스키 미터법(Minkowski)의 차수를 결정한다. 
	# p = 1이면 맨해튼 거리(Manhatten distance)
	# p = 2이면 유클리드 거리(Euclidean distance)이다. 

<br>

## k-최근접 이웃 분류(k-Nearest Neighbors Classification)
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

	from sklearn.neighbors import KNeighborsClassifier
	kn = KNeighborsClassifier()

	#훈련
	kn.fit(train_input, train_target)
	#평가
	print(kn.score(test_input, test_target))

<br>

<br>

# [4-2] 서포트 벡터 머신(Support Vector Machine, SVM)
▣ 정의 : 데이터를 고차원 공간으로 매핑한 후, 클래스 간 마진(여유폭)을 최대화하는 초평면(hyperplane)을 찾아 분류 혹은 회귀하는 지도학습 기법<br> 
N차원 공간을 (N-1)차원으로 나눌 수 있는 초평면을 찾는 분류 기법으로 2개의 클래스를 분류할 수 있는 최적의 경계를 찾는다.<br>
▣ 목적 : 특히 경계가 선형이 아니거나, 고차원 공간에서 마진이 중요한 문제에 대해 강건한 분류/회귀 모델을 구축<br>
▣ 장점 : 마진 최대화라는 견고한 이론 기반, 커널을 사용해 비선형 데이터도 효과적으로 처리, 고차원 특성 공간에서 비교적 잘 작동<br>
▣ 단점 : 훈련 및 예측 시간이 샘플 수 및 특성 수에 따라 급격히 증가, 커널 선정·하이퍼파라미터 튜닝과 결과 해석이 까다롭다<br>
▣ Scikit-learn 클래스명 : 분류용 sklearn.svm.SVC 회귀용 sklearn.svm.SVR<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html<br>
▣ API : https://scikit-learn.org/stable/auto_examples/svm/index.html<br>

![](./images/margin.png)

**최적의 경계 :** 각 클래스의 말단에 위치한 데이터들 사이의 거리를 최대화 할 수 있는 경계<br>
**초평면(hyper plane) :** 고차원(N차원)에서 데이터를 두 분류로 나누는 결정 경계<br>
**Support Vector :** 데이터들 중에서 결정 경계에 가장 가까운 데이터들<br>
**마진(Margin) :** 결정 경계와 support vector사이의 거리<br>
**비용(Cost) :** 마진(Margin) 크기의 반비례<br>
**감마(Gamma) :** train data 하나 당 결정 경계에 영향을 끼치는 범위를 조절하는 변수(크면 오버피팅, 작으면 언더피팅)<br>


| 장점                             | 단점                                              |
|----------------------------------|---------------------------------------------------|
| 과적합을 피할 수 있다 | 커널함수 선택이 명확하지 않다 |
| 분류 성능이 좋다 | 파라미터 조절을 적절히 수행해야만 최적의 모델을 찾을 수 있다 |
| 저차원, 고차원 공간의 적은 데이터에 대해서 일반화 능력이 우수 | 계산량 부담이 있다 |
| 잡음에 강하다 | 데이터 특성의 스케일링에 민감하다|
| 데이터 특성이 적어도 좋은 성능 | | 

▣ 유형 : 선형SVM(하드마진, 소프트마진), 비선형SVM<br>
- 하드마진 : 두 클래스를 분류할 수 있는 최대마진의 초평면을 찾는 방법으로, 모든 훈련데이터는 마진의 바깥족에 위치하게 선형으로 구분해서 하나의 오차도 허용하면 안된다. 모든 데이터를 선형으로 오차없이 나눌 수 있는 결정경계를 찾는 것은 사실상 어렵다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|\left\|w\right\|\right\|^2$     제약 조건은 모든 i에 대해 $𝑦_𝑖(𝑤⋅𝑥_𝑖+𝑏)≥1$ <br>

![](./images/hmargin.png)

- 소프트마진 :  하드마진이 가진 한계를 개선하고자 나온 개념으로, 완벽하게 분류하는 초평면을 찾는 것이 아니라 어느 정도의 오분류를 허용하는 방식이다. 소프트마진에서는 오분류를 허용하고 이를 고려하기 위해 slack variable을 사용하여 해당 결정경계로부터 잘못 분류된 데이터의 거리를 측정한다.<br><br>
$\displaystyle \min_{w}\frac{1}{2}\left\|\left\|w\right\|\right\|^2 + C\sum_{i=1}^{n}\xi_i$

![](./images/smargin.png)

- 비선형분류 : 선형분리가 불가능한 입력공간을 선형분리가 가능한 고차원 특성공간으로 보내 선형분리를 진행하고 그 후 다시 기존의 입력공간으로 변환하면 비선형 분리를 하게 된다.<br><br>
![](./images/nlsvm.png)

<br>

	from sklearn.datasets import make_moons
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import PolynomialFeatures

	polynomial_svm_clf = 
	Pipeline([("poly_features", PolynomialFeatures(degree=3)),("scaler", StandardScaler()),
			("svm_clf", LinearSVC(C=10, loss="hinge", max_iter=2000, random_state=42))])
	polynomial_svm_clf.fit(X, y)

<br>

입력공간을 특성공간으로 변환하기 위해서 mapping function을 사용한다<br>
$\Phi(x) = Ax$<br><br>
고차원의 특성공간으로 변환하고 목적함수에 대한 문제를 푸는 것이 간단한 차원에서는 가능하나 그 차수가 커질수록 계산량의 증가하는 것을 다시 해결하고자 나오는 개념이 커널트릭(Kernel trick) : 비선형 분류를 하기 위해 차원을 높여줄 때마다 필요한 엄청난 계산량을 줄이기 위해 사용하는 커널 트릭은 실제로는 데이터의 특성을 확장하지 않으면서 특성을 확장한 것과 동일한 효과를 가져오는 기법<br>
$k(x_i, x_j) =\Phi(x_i)^T\Phi(x_j)$<br><br>
확장된 특성공간의 두 벡터의 내적만을 계산하여 고차원의 복잡한 계산 없이 커널 함수를 사용하여 연산량을 간단하게 해결할 수 있다. 가장 성능이 좋고 많이 사용되는 것이 가우시안 RBF(Radial basis function)으로 두 데이터 포인트 사이의 거리를 비선형 방식으로 변환하여 고차원 특징 공간에서 분류 문제를 해결하는 데 사용.<br><br>
$k(x,y) = e^{-\frac{-\left\|x_i-x_j\right\|^2}{2\sigma^2}}$<br><br>
직접 차수를 정하는 방식(Polynomial) : $k(x,y) = (1+x^Ty)^p$<br>
신경망 학습(Signomail) : $k(x,y) = tanh(kx_ix_j-\delta)$<br>

![](./images/hnlsvm.png)

<br>  

## 서포트 벡터 회귀(Support Vector Regression, SVR)
▣ 정의 : 데이터 포인트들을 초평면 근처에 배치하면서, 허용 오차 $ϵ$ 내에서 예측 오차를 최소화하는 것.<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#regression<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html<br>
▣ 예제 : https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html<br>
▣ 모델식 : https://scikit-learn.org/stable/modules/svm.html#svr<br>


	from sklearn.svm import SVR
 
 	svr = SVR(kernel='rbf', gamma='auto')
	svr.fit(xtrain, ytrain)

	score = svr.score(xtest, ytest)
	print("R-squared: ", score)

<br> 

## 서포트 벡터 분류(Support Vector Classification, SVC)
▣ 정의 : 두 클래스(또는 다수의 클래스)를 분류하기 위해 최대 마진을 가지는 초평면을 찾는 것.<br>
▣ 가이드 : https://scikit-learn.org/stable/modules/svm.html#classification<br>
▣ API : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC<br>
▣ 모델식 : https://scikit-learn.org/stable/modules/svm.html#svc<br>


	import sklearn.svm as svm

 	# 선형일 경우
	svm_clf =svm.SVC(kernel = 'linear')
 	# 비선형일 경우
 	svm_clf =svm.SVC(kernel = 'rbf')

	# 교차검증
	scores = cross_val_score(svm_clf, X, y, cv = 5)
 	scores.mean()


<br>




| 모델                                                         | 수식                                                                                                                                                                                                                                                      | 주요 적용 분야                           |
| :------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------- |
| **[1-1] LDA (Linear Discriminant Analysis)**                   | ![lda](https://latex.codecogs.com/svg.image?%5Cdelta_k%28x%29%3Dx%5E%5Ctop%20%5CSigma%5E%7B-1%7D%5Cmu_k-%5Ctfrac12%20%5Cmu_k%5E%5Ctop%20%5CSigma%5E%7B-1%7D%5Cmu_k%2B%5Clog%20%5Cpi_k)                                                                     | 다중 클래스 분류, 얼굴 인식, 문서 분류, 의료 데이터(당뇨환자의 혈액데이터 분류)    |
| **[1-2] QDA (Quadratic Discriminant Analysis)**                | ![qda](https://latex.codecogs.com/svg.image?%5Cdelta_k%28x%29%3D-%5Ctfrac12%20%5Clog%7C%5CSigma_k%7C-%5Ctfrac12%20%28x-%5Cmu_k%29%5E%5Ctop%20%5CSigma_k%5E%7B-1%7D%20%28x-%5Cmu_k%29%2B%5Clog%20%5Cpi_k)                                                   | 공분산이 클래스별로 다른 분류, 생물정보, 금융 리스크(신용카드 부정거래 탐지)     |
| **[2-1] PCR (Principal Component Regression)**                 | ![pcr](https://latex.codecogs.com/svg.image?Z%3DXW_%7BPCA%7D%2C%20%5Chat{y}%3DZ%5Chat{%5Cbeta})                                                                                                                                                            | 다중공선성 완화 회귀, 스펙트럼 분석, 공정 데이터 예측(반도체 공정 결함 예측)    |
| **[2-2] PLS (Partial Least Squares)**                          | ![pls](https://latex.codecogs.com/svg.image?X%3DTP%5E%5Ctop%2BE%2C%20Y%3DUQ%5E%5Ctop%2BF%2C%20%5Cmax%20%5Coperatorname%7BCov%7D%28T%2CU%29)                                                                                                                | X–Y 상관이 높은 예측, 화학계량학, 공정 모니터링(생산라인 품질관리)      |
| **[2-3] PLS-DA (Partial Least Squares Discriminant Analysis)** | ![plsda](https://latex.codecogs.com/svg.image?Y%5Cin%7B0%2C1%2C%5Cdots%7D%2C%20T%3DXW%2C%20%5Cmax%20%5Coperatorname%7BCov%7D%28T%2CY%29)                                                                                                                   | 다중 클래스 분류, 오믹스 분석, 품질 검사, 바이오마커 탐색(암 단백질체 데이터 분석) |
| **[2-4] Supervised PCA**                                       | ![spca](https://latex.codecogs.com/svg.image?%5Cmax_%7Bw%3A%5C%7Cw%5C%7C%3D1%7D%20%5Coperatorname%7BCorr%7D%28Xw%2C%20y%29)                                                                                                                                | 라벨 정보 활용 차원축소, 이미지/텍스트 분류 전처리(감정 분류, 뉴스기사 주제 분류)      |
| **[3-1] 결정 트리 (Decision Tree)**                                | ![tree](https://latex.codecogs.com/svg.image?I%28t%29%3D-%5Csum_i%20p_i%28t%29%5Clog%20p_i%28t%29)                                                                                                                                                         | 분류·회귀, 변수 중요도 분석(고객 이탈 예측)            |
| **[3-2] 랜덤 포레스트 (Random Forest)**                              | ![rf](https://latex.codecogs.com/svg.image?%5Chat{y}%28x%29%3D%5Ctfrac1B%20%5Csum_%7Bb%3D1%7D%5EB%20h_b%28x%29)                                                                                                                                            | 대규모 분류·회귀, 변수 중요도, 이상 탐지(신용 평가, 센서 데이터 이상탐지)           |
| **[4-1] K-NN (k-Nearest Neighbors)**                           | 분류: ![knn1](https://latex.codecogs.com/svg.image?%5Chat{y}%3D%5Coperatorname%7Bmode%7D%5C%7By_i%3A%20x_i%5Cin%20N_k%28x%29%5C%7D) <br> 회귀: ![knn2](https://latex.codecogs.com/svg.image?%5Chat{y}%3D%5Ctfrac1k%20%5Csum_%7Bx_i%5Cin%20N_k%28x%29%7D%20y_i) | 패턴 인식, 추천 시스템, 비모수 근접 예측(사용자 취향기반 영화 추천)           |
| **[4-2] SVM (Support Vector Machine)**                         | ![svm](https://latex.codecogs.com/svg.image?%5Cmin_%7Bw%2Cb%7D%20%5Ctfrac12%20%5C%7Cw%5C%7C%5E2%20%5Ctext%7Bs.t.%7D%20%20y_i%28w%5E%5Ctop%20x_i%2Bb%29%5Cge%201)                                                                                           | 이진/다중 분류, 고차원 텍스트/이미지, 생체 신호(얼굴 감정 인식)      |





 
![](./images/SLC.png)
<br>출처 : https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
