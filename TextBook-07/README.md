
![](./images/NSML.png)

#  07-1 : 군집화 평가지표

	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합
	  

#  07-2 : 연관규칙 평가지표

	[1] 지지도(Support)
	[2] 신뢰도(Confidence)
	[3] 향상도(Lift)
	[4] 레버리지(Leverage)
	[5] 확신도(Conviction)
	[6] 상관계수(Correlation Coefficient)
	  
#  07-3 : 차원축소 평가지표

	[1] 재구성 오류(Reconstruction Error)
	[2] 분산 유지율(Explained Variance Ratio)
	[3] 상호 정보량(Mutual Information)
	[4] 근접도 보존(Trustworthiness, Continuity)
	[5] 거리/유사도 보존(Stress, Sammon Error)
	[6] 지역/전역구조(Local Continuity Meta Criterion)
	[7] 쌍의 상관계수(Spearman’s ρ)
	[8] Silhouette Score
	[9] Davies-Bouldin Index(DBI)
	[10] Adjusted Rand Index(ARI)
	[11] Normalized Mutual Information(NMI)


---

#  07-1 : 군집화 평가지표

---

	[1] Silhouette Coefficient : 실루엣 계수
	[2] Davies-Bouldin Index (DBI)
	[3] Dunn Index (DI)
	[4] Calinski-Harabasz Index (CHI)
	[5] Within-Cluster Sum of Squares (WCSS) : 군집내 제곱합
	  
---


## ▣ 군집화 평가지표 수식

| 지표 | 의미 | 수식 |
|---|---|---|
| **[1] Silhouette Coefficient** | 한 점이 자기 군집 평균거리 $a(i)$ 대비 가장 가까운 다른 군집 평균거리 $b(i)$ 로 분리도 측정 | $s(i)=\dfrac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}$<br>$\bar{s}=\dfrac{1}{n}\sum_{i=1}^{n}s(i)$ |
| **[2] Davies–Bouldin Index (DBI)** | 군집 내 응집도 대비 군집 간 분리도 | $DBI=\dfrac{1}{K}\sum_{i=1}^{K}\max_{j\ne i}\dfrac{S_i+S_j}{M_{ij}}$<br>$S_i=\dfrac{1}{\lvert C_i\rvert}\sum_{x\in C_i}\lVert x-\mu_i\rVert,\; M_{ij}=\lVert\mu_i-\mu_j\rVert$ |
| **[3] Dunn Index (DI)** | 가장 가까운 군집 간 최소거리 대비 최대 군집 지름 | $DI=\dfrac{\min_{i\ne j}\,\delta(C_i,C_j)}{\max_k\,\Delta(C_k)}$<br>$\delta$: 군집 간 최소거리, $\Delta$: 군집 지름(내부 최대거리) |
| **[4] Calinski–Harabasz Index (CHI)** | 군집 사이 분산 / 군집 내 분산 | $CH=\dfrac{\mathrm{Tr}(B_K)/(K-1)}{\mathrm{Tr}(W_K)/(n-K)}$<br>$\mathrm{Tr}(B_K)=\sum_k \lvert C_k\rvert\,\lVert\mu_k-\bar{x}\rVert^2,\;\mathrm{Tr}(W_K)=\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$ |
| **[5] Within-Cluster Sum of Squares (WCSS)** | 각 점이 군집 중심까지의 제곱거리 합 K-means 목적함수와 동일 | $\mathrm{WCSS}=\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k\rVert^{2}$ |


<br>


## ▣ 군집화 평가지표 결과해석

| 지표                                | 목표                    | 권장 해석 기준                                                           | 비고                                                             |
| --------------------------------- | --------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------- |
| **[1] Silhouette Coefficient**   | ↑(−1~1)   | **≥ 0.70 우수**, **0.50~0.69 양호**, **0.25~0.49 보통**, **< 0.25 미흡** | 군집 내 응집 vs 인접 군집과 분리. 평균값과 군집별 분포를 함께 확인 권장.               |
| **[2] Davies–Bouldin Index (DBI)**  | `↓`(하한 0) | **≤ 0.50 우수**, **0.51~0.99 양호**, **1.00~1.49 보통**, **≥ 1.50 미흡** | 군집 응집도 대비 중심 간 분리. 스케일·거리척도에 민감.  |
| **[3] Dunn Index (DI)**   | ↑(상한 데이터 의존) | **≥ 0.50 우수**, **0.30~0.49 양호**, **0.10~0.29 보통**, **< 0.10 미흡** | 최소 군집 간 거리 / 최대 지름이라 값이 전반적으로 작게 나오는 편. 이상치·밀도 차이에 민감. |
| **[4] Calinski–Harabasz Index (CHI)** | ↑ | **절대 임계치 없음** → 동일 데이터에서 k 간 상대 비교: 국소 최대/엘보우 지점이 바람직           | 데이터/스케일에 강하게 의존. 보통 k 스윕으로 비교.                             |
| **[5] Within-Cluster Sum of Squares (WCSS)**    | `↓` | **절대 임계치 없음** → 엘보우 지점에서 k 선택                                 | 표준화 여부·특징 개수에 좌우. k↑→단조감소이므로 상대 비교 전용.                 |


<br>


### Iris 데이터 + K-means(k=3) 학습 후 평가지표 5종 출력 소스

**[1] Silhouette Coefficient**
<br>
**[2] Davies-Bouldin Index(DBI)**
<br>
**[3] Dunn Index(DI)** scikit-learn 미제공으로 커스텀 구현
<br>
**[4] Calinski-Harabasz Index(CHI)**
<br>
**[5] Within-Cluster Sum of Squares(WCSS = inertia)**
<br>


	# ---------- 경고 방지용 환경변수: 반드시 상단에서 설정 ----------
	import os
	os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP 스레드
	os.environ["MKL_NUM_THREADS"] = "1"       # MKL 스레드
	os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS 스레드
	os.environ["NUMEXPR_NUM_THREADS"] = "1"   # numexpr 스레드
	# 물리 코어 수 직접 지정하고 싶다면 주석 해제 (예: 8코어)
	# os.environ["LOKY_MAX_CPU_COUNT"] = "8"
	
	# ------------------------- 라이브러리 임포트 -------------------------
	import numpy as np
	from sklearn import datasets
	from sklearn.cluster import KMeans
	from sklearn.metrics import (
	    silhouette_score,
	    davies_bouldin_score,
	    calinski_harabasz_score,
	)
	from sklearn.metrics import pairwise_distances
	
	
	# ------------------------- Dunn Index (커스텀) -------------------------
	def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
	    """
	    Dunn Index (값이 클수록 좋음)
	    DI = (군집 간 최소거리) / (군집 내 최대 지름)
	    - 단일 샘플 군집이 있거나, 분모가 0이면 NaN 반환
	    """
	    unique_labels = np.unique(labels)
	    clusters = [X[labels == c] for c in unique_labels]
	
	    if len(clusters) < 2:
	        return float("nan")
	
	    # 군집 내 최대 지름(최대 쌍거리)
	    intra_max = 0.0
	    for c in clusters:
	        if len(c) <= 1:
	            # 단일 포인트 군집은 지름이 정의 어려움 → 넘어가되 전체 판단에 영향
	            continue
	        d = pairwise_distances(c)
	        intra_max = max(intra_max, float(np.max(d)))
	
	    # 군집 간 최소거리(단일 링크)
	    inter_min = np.inf
	    for i in range(len(clusters)):
	        for j in range(i + 1, len(clusters)):
	            if len(clusters[i]) == 0 or len(clusters[j]) == 0:
	                continue
	            d = pairwise_distances(clusters[i], clusters[j])
	            inter_min = min(inter_min, float(np.min(d)))
	
	    if intra_max == 0.0 or not np.isfinite(inter_min):
	        return float("nan")
	    return inter_min / intra_max
	
	
	# ------------------------- 메인 루틴 -------------------------
	def main(k: int = 3, random_state: int = 42) -> None:
	    # 1) 데이터 로드
	    iris = datasets.load_iris()
	    X = iris.data
	
	    # 2) K-means 학습
	    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
	    labels = km.fit_predict(X)
	
	    # 3) 지표 계산
	    silhouette = float(silhouette_score(X, labels))
	    dbi = float(davies_bouldin_score(X, labels))
	    dunn = float(dunn_index(X, labels))
	    chi = float(calinski_harabasz_score(X, labels))
	    wcss = float(km.inertia_)  # Within-Cluster Sum of Squares
	
	    # 4) 출력
	    print(f"=== Iris + K-means (k={k}) ===")
	    print(f"[1] Silhouette Coefficient : {silhouette:.4f}")
	    print(f"[2] Davies-Bouldin Index   : {dbi:.4f}")
	    print(f"[3] Dunn Index             : {dunn:.4f}")
	    print(f"[4] Calinski-Harabasz     : {chi:.4f}")
	    print(f"[5] WCSS (Inertia)        : {wcss:.4f}")
	
	
	if __name__ == "__main__":
	    main(k=3)



### (소스 실행 결과)

	=== Iris + K-means (k=3) ===
	[1] Silhouette Coefficient : 0.5528
	[2] Davies-Bouldin Index   : 0.6620
	[3] Dunn Index             : 0.0988
	[4] Calinski-Harabasz     : 561.6278
	[5] WCSS (Inertia)        : 78.8514


### (결과 분석)

	[1] Silhouette Coefficient : 0.5528		=> 0.5 이상이면 양호(군집 내 응집도와 군집 간 분리가 꽤 확보됨)
	[2] Davies-Bouldin Index   : 0.6620		=> 1 미만이면 보통(분리도/응집도 균형이 준수)
	[3] Dunn Index             : 0.0988		=> 1 미만이면 미흡(군집 간 최소거리가 작게 잡혀 낮게 나왔을 가능성)
	[4] Calinski-Harabasz     : 561.6278	=> k=3이 꽤 설득력 있음
	[5] WCSS (Inertia)        : 78.8514		=> 응집도 양호(단독 평가는 어렵고, k를 바꿔 엘보우로 비교 추천)

---

#  07-2 : 연관규칙 평가지표

---
	
	[1] 지지도(Support)
	[2] 신뢰도(Confidence)
	[3] 향상도(Lift)
	[4] 레버리지(Leverage)
	[5] 확신도(Conviction)
	[6] 상관계수(Correlation Coefficient)
	  
---

## ▣ 연관규칙 평가지표 수식
           
| 지표                       | 의미                                         | 수식                                                             |
| ------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| **[1] 지지도(Support)**         | A와 B가 동시에 발생할 비율                       | `support(A→B) = P(A ∧ B) = count(A ∧ B) / N`                           |
| **[2] 신뢰도(Confidence)**      | A가 발생했을 때 B가 함께 발생할 조건부 확률             | `confidence(A→B) = P(B ∣ A) = P(A ∧ B) / P(A)`                         |
| **[3] 향상도(Lift)**            | 독립 가정 대비 연관 강도                         | `lift(A→B) = P(A ∧ B) / ( P(A) · P(B) ) = confidence(A→B) / P(B)`      |
| **[4] 레버리지(Leverage)**       | 실제 동시발생과 기대 동시발생의 차이                   | `leverage(A,B) = P(A ∧ B) − P(A)·P(B)`                                 |
| **[5] 확신도(Conviction)**      | 규칙이 없을 때의 B 부정 확률 대비, 규칙 하의 오류율 감소 | `conviction(A→B) = (1 − P(B)) / (1 − confidence(A→B))`                 |
| **[6] 상관계수(Correlation)** | A–B 이진 상관(피어슨 φ)                       | `φ(A,B) = ( P(A ∧ B) − P(A)·P(B) ) / √( P(A)(1−P(A)) · P(B)(1−P(B)) )` |


<br>


## ▣ 연관규칙 평가지표 결과해석

| 지표            | 목표            | 권장 해석 기준            | 비고                       |  
| :-------------- | :------------- | :----------------------- | :------------------------ | 
| **[1] 지지도 (Support)**    | 전체 거래 중 A와 B가 함께 등장하는 비율<br>→ 규칙의 빈도·보편성 평가   | 값이 높을수록 빈발한 규칙<br>보통 `0.01~0.05` 이상이면 의미 있음 (도메인에 따라 다름)   | 빈도 기반 필터로 먼저 사용됨<br>너무 낮으면 희귀, 너무 높으면 일반 규칙           |  
| **[2] 신뢰도 (Confidence)**  | A가 발생했을 때 B도 발생할 조건부 확률<br>→ 규칙의 정확도 평가   | 값이 1에 가까울수록 강한 규칙<br>보통`0.6~0.9` 이상이면 신뢰 높은 규칙 | 단독 빈도가 높은 B항목은 과대평가 가능 |
| **[3] 향상도 (Lift)**  | A와 B가 독립일 때 대비 얼마나 자주 함께 발생하는가<br>→ 상관성 강도 평가 | >1 → 양의 상관<br> =1 → 독립<br> <1 → 음의 상관 | 신뢰도의 편향 보정 지표<br>가장 자주 사용하는 상관성 지표   | 
| **[4] 레버리지 (Leverage)**  | 독립 가정 하 기대빈도와 실제빈도의 차이<br>→ A,B의 공동발생 초과 정도   | 양수 → 양의 상관<br> 0 → 독립<br> 음수 → 음의 상관 | 확률 차이 자체를 표현해 직관적<br>값의 범위: `[-0.25, +0.25]` 내외|    
| **[5] 확신도 (Conviction)**  | “A 발생 시 B가 발생하지 않을 확률”의 역수<br>→ 규칙의 방향성·일관성 평가    | =1 → 독립<br> >1 → 긍정적 관계<br> <1 → 음의 관계   | Lift와 달리 A→B 방향성 고려<br>신뢰도와 함께 해석 시 유용   |     
| **[6] 상관계수 (Correlation Coefficient, φ)** | 두 사건 A, B 간의 선형 상관 정도<br>→ 동시발생의 독립성 측정  | +1 → 완전 양의 상관<br> 0 → 독립<br> -1 → 완전 음의 상관 | Lift·Leverage와 유사하지만 정규화되어 있음<br>확률 분포 불균형에 민감   |   


### Groceries 데이터 + Apriori 학습 후 평가지표 6종 출력 소스

**[1] 지지도(Support)**
<br>
**[2] 신뢰도(Confidence)**
<br> 
**[3] 향상도(Lift)**
<br> 
**[4] 레버리지(Leverage)**
<br> 
**[5] 확신도(Conviction)**
<br> 
**[6] 상관계수(Correlation Coefficient)**
<br>
	
	
	import os
	import io
	import warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning)

	import numpy as np
	import pandas as pd
	from mlxtend.preprocessing import TransactionEncoder
	from mlxtend.frequent_patterns import apriori, association_rules

	# -------------------------------
	# 0) 설정 (필요 시 조정)
	# -------------------------------
	DEFAULT_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
	LOCAL_FILE  = "groceries.csv"   # 오프라인 사용 시 동일 파일명을 같은 폴더에 두세요
	
	# 채굴 파라미터 (규칙이 너무 적으면 min_support/min_confidence를 더 낮추세요)
	MIN_SUPPORT    = 0.005   # 0.5% 이상 거래에서 등장하는 항목집합
	MIN_CONFIDENCE = 0.10    # 10% 이상 신뢰도
	PAIR_RULE_ONLY = True    # True면 X -> Y 한 항목씩(2-아이템 규칙)만 출력

	# 품질 필터 (원치 않으면 False로)
	FILTER_LIFT_GT      = 1.05   # 향상도 > 1.05 (독립보다 의미 있게 높음)
	FILTER_LEVERAGE_POS = True   # 레버리지 > 0 (양의 연관만)

	# -------------------------------
	# 1) Groceries 로더 (견고 모드)
	#    - 행마다 필드 개수 상이 → python 엔진/폴백 파싱
	# -------------------------------
	def robust_read_csv(path_or_buf):
   	 """유연한 CSV 파서: 실패 시 raw 텍스트 라인 스플릿 폴백."""
    	try:
       		 df = pd.read_csv(
            	path_or_buf,
            	header=None,
           	 	engine="python",
            	sep=",",
            	quotechar='"',
            	skipinitialspace=True,
            	on_bad_lines="skip",
       	 )
        	return df
    	except Exception as e:
        	# raw 텍스트 파싱
        	try:
            	if isinstance(path_or_buf, str) and path_or_buf.startswith("http"):
                	import urllib.request
               	 	with urllib.request.urlopen(path_or_buf) as resp:
                    	raw = resp.read().decode("utf-8", errors="ignore")
            	elif isinstance(path_or_buf, str):
                	with open(path_or_buf, "r", encoding="utf-8", errors="ignore") as f:
                    	raw = f.read()
            	else:
                	raw = path_or_buf.read()
                	if isinstance(raw, bytes):
                    	raw = raw.decode("utf-8", errors="ignore")

            	rows = []
            	for line in raw.splitlines():
                	line = line.strip()
                	if not line:
                    	continue
                	items = [x.strip().strip('"').strip("'") for x in line.split(",") if x.strip()]
                	rows.append(items)
            	max_len = max((len(r) for r in rows), default=0)
            	rows_pad = [r + [None]*(max_len - len(r)) for r in rows]
            	return pd.DataFrame(rows_pad)
        	except Exception as e2:
            	raise RuntimeError(f"Groceries CSV 파싱 실패: {e}\nFallback 오류: {e2}")

	def load_groceries():
    	# 로컬 우선, 없으면 URL
    	if os.path.exists(LOCAL_FILE):
        	df = robust_read_csv(LOCAL_FILE)
    	else:
        	df = robust_read_csv(DEFAULT_URL)

    	# 각 행 → 아이템 리스트(트랜잭션)
    	transactions = []
    	for _, row in df.iterrows():
        	items = [str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip() != ""]
       		# 중복 제거
        	items = list(dict.fromkeys(items))
       		if items:
            	transactions.append(items)
    	return transactions

	# -------------------------------
	# 2) 데이터 로드 & 원-핫 인코딩
	# -------------------------------
	transactions = load_groceries()
	print(f"[INFO] Loaded transactions: {len(transactions):,}")

	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	basket = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)
	print(f"[INFO] Unique items: {basket.shape[1]:,}")

	# -------------------------------
	# 3) Apriori → 빈발 항목집합
	# -------------------------------
	frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)

	# itemset → support 맵 (버전 호환 및 φ 계산용)
	support_map = {
    	frozenset(items): supp
    	for items, supp in zip(frequent_itemsets['itemsets'], frequent_itemsets['support'])
	}

	# -------------------------------
	# 4) 연관규칙 + 지표 계산
	# -------------------------------
	rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

	# 구버전 호환: antecedent/consequent support 없으면 채움
	if 'antecedent support' not in rules.columns:
    	rules['antecedent support'] = rules['antecedents'].apply(lambda A: support_map.get(frozenset(A), np.nan))
	if 'consequent support' not in rules.columns:
    	rules['consequent support'] = rules['consequents'].apply(lambda B: support_map.get(frozenset(B), np.nan))

	# (선택) 2-아이템 규칙만 (X→Y에서 X, Y 각각 단일 아이템)
	if PAIR_RULE_ONLY:
    	rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]

	# 상관계수(φ) = (P(XY) - P(X)P(Y)) / sqrt(P(X)(1-P(X)) P(Y)(1-P(Y)))
	def phi_coef(p_xy, p_x, p_y, eps=1e-12):
    	num = p_xy - (p_x * p_y)
    	den = np.sqrt(max(p_x * (1 - p_x) * p_y * (1 - p_y), 0.0)) + eps
    	return num / den

	rules['phi'] = rules.apply(
    	lambda r: phi_coef(
        	p_xy=r['support'],
        	p_x=r['antecedent support'],
        	p_y=r['consequent support']
    	),
    	axis=1
	)

	# 품질 필터 적용
	if FILTER_LIFT_GT is not None:
    	rules = rules[rules['lift'] > FILTER_LIFT_GT]
	if FILTER_LEVERAGE_POS:
    	rules = rules[rules['leverage'] > 0]

	# -------------------------------
	# 5) 출력 정리
	# -------------------------------
	def items_to_str(s):
    	return ", ".join(sorted(map(str, s)))

	cols_out = [
    	'antecedents', 'consequents',
    	'support', 'confidence', 'lift', 'leverage', 'conviction', 'phi'
	]
	view = rules[cols_out].copy() if len(rules) else pd.DataFrame(columns=cols_out)
	if len(view):
    	view['antecedents'] = view['antecedents'].apply(items_to_str)
    	view['consequents'] = view['consequents'].apply(items_to_str)
    	view = view.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)

	pd.set_option('display.max_colwidth', 120)
	print("\nGroceries + Apriori Association Rules (sorted by lift)\n" + "-"*90)
	print(view.to_string(index=False) if len(view) else "(no rules)")

	print("\n[Summary] #rules:", len(view),
    	  "| support∈[%.4f, %.4f]" % (view['support'].min() if len(view) else 0,
    	                              view['support'].max() if len(view) else 0),
     	 "| confidence∈[%.4f, %.4f]" % (view['confidence'].min() if len(view) else 0,
           	                          view['confidence'].max() if len(view) else 0))
	

### (소스 실행 결과)

	Groceries + Apriori Association Rules (sorted by lift)
	------------------------------------------------------------------------------------------
	antecedents      					consequents  support  confidence     lift  leverage  conviction      phi
	liquor     bottled beer 			0.005896    0.562500 8.459667  0.005199    2.133733 0.204901
	root vegetables other vegetables 	0.007534    0.202643 2.330206  0.004301    1.145079 0.080669
	frankfurter       rolls/buns 		0.008189    0.284091 2.252804  0.004554    1.220678 0.081988
	sausage       rolls/buns 			0.013429    0.278912 2.211732  0.007358    1.211910 0.103527
	rolls/buns          sausage 		0.013429    0.106494 2.211732  0.007358    1.065298 0.103527
	citrus fruit other vegetables 		0.006060    0.177885 2.045506  0.003097    1.110594 0.060594
	curd       whole milk 				0.005405    0.266129 1.891716  0.002548    1.170940 0.051946
	root vegetables       whole milk 	0.007861    0.211454 1.503069  0.002631    1.089751 0.039997
	brown bread       whole milk 		0.006387    0.200000 1.421653  0.001894    1.074148 0.030987
	bottled water             soda 		0.010809    0.162562 1.284089  0.002391    1.042946 0.028866
	pastry       whole milk 			0.008844    0.175325 1.246254  0.001747    1.042009 0.022965
	pastry             soda 			0.007861    0.155844 1.231028  0.001475    1.034647 0.020272
	tropical fruit       whole milk 	0.006715    0.168724 1.199337  0.001116    1.033735 0.016420
	other vegetables       whole milk 	0.014576    0.167608 1.191404  0.002342    1.032349 0.023901
	whole milk other vegetables 		0.014576    0.103609 1.191404  0.002342    1.018569 0.023901
	newspapers       whole milk 		0.008025    0.165541 1.176706  0.001205    1.029791 0.016138
	sausage             soda 			0.006878    0.142857 1.128442  0.000783    1.018970 0.010998
	rolls/buns             soda 		0.017851    0.141558 1.118183  0.001887    1.017429 0.017092
	soda       rolls/buns 				0.017851   	0.141009 1.118183  0.001887    1.017350 0.017092
	shopping bags             soda 		0.006715    0.138514 1.094131  0.000578    1.013833 0.008089

	[Summary] #rules: 20 | support∈[0.0054, 0.0179] | confidence∈[0.1036, 0.5625]


### (결과 분석)


	Apriori 알고리즘이 설정된 지지도(support)·신뢰도(confidence) 기준을 만족하는 20개의 규칙(X→Y)을 발견
	이 규칙들은 전체 거래의 0.5~1.8%에서 등장했고,
	조건부 확률(신뢰도)은 10%~56%로,
	일부는 현실적으로 꽤 강한 구매 연관성을 가진다는 의미

---

#  07-3 : 차원축소 평가지표

---
	
	▣ 재구성 기반 : 원본 복원 능력
	[1] 재구성 오류(Reconstruction Error)
	[2] 분산 유지율(Explained Variance Ratio)
	[3] 상호 정보량(Mutual Information)

	▣ 구조 보존 기반 : 거리·이웃 관계 유지
	[4] 근접도 보존(Trustworthiness, Continuity)
	[5] 거리/유사도 보존(Stress, Sammon Error)
	[6] 지역/전역구조(Local Continuity Meta Criterion)
	[7] 쌍의 상관계수(Spearman’s ρ)

	▣ 활용 성능 기반 : 축소된 표현의 유용성
	[8] Silhouette Score
	[9] Davies-Bouldin Index(DBI)
	[10] Adjusted Rand Index(ARI)
	[11] Normalized Mutual Information(NMI)
	  
---

## ▣ 차원축소 평가지표 수식

| 지표명 | 수식 | 설명 |
|---------|------|------|
| **[1] 재구성 오류 (Reconstruction Error)** | ![](https://latex.codecogs.com/svg.image?RE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5C%7C%20x_i%20-%20%5Chat%7Bx%7D_i%20%5C%7C%5E2) | 원본 데이터와 복원된 데이터의 평균제곱오차(MSE). 값이 작을수록 복원력이 높음. |
| **[2] 분산 유지율 (Explained Variance Ratio)** | ![](https://latex.codecogs.com/svg.image?EVR_k%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Clambda_i%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Clambda_i%7D) | 상위 k개의 고유값이 전체 분산에서 차지하는 비율. PCA 등에서 정보 손실 정도 평가. |
| **[3] 상호 정보량 (Mutual Information)** | ![](https://latex.codecogs.com/svg.image?MI(X%2CY)%20%3D%20%5Csum_%7Bx%20%5Cin%20X%7D%20%5Csum_%7By%20%5Cin%20Y%7D%20p(x%2Cy)%5Clog%5Cfrac%7Bp(x%2Cy)%7D%7Bp(x)p(y)%7D) | 축소 전후 데이터의 정보량 비교. 값이 클수록 정보 보존이 잘됨. |
| **[4-1] 근접도 보존 – Trustworthiness** | ![](https://latex.codecogs.com/svg.image?T(k)%20%3D%201%20-%20%5Cfrac%7B2%7D%7Bnk(2n-3k-1)%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%20%5Cin%20U_k(i)%7D%20(r(i%2Cj)%20-%20k)) | 고차원에서 이웃이 아니던 점이 저차원에서 잘못 가까워지는 정도를 측정. |
| **[4-2] 근접도 보존 – Continuity** | ![](https://latex.codecogs.com/svg.image?C(k)%20%3D%201%20-%20%5Cfrac%7B2%7D%7Bnk(2n-3k-1)%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%20%5Cin%20V_k(i)%7D%20(r'(i%2Cj)%20-%20k)) | 저차원에서 이웃이던 점이 고차원에서 멀어지는 정도를 측정. |
| **[5-1] 거리 보존 – Stress (Kruskal’s Stress)** | ![](https://latex.codecogs.com/svg.image?Stress%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3Cj%7D(d_%7Bij%7D-%5Chat%7Bd%7D_%7Bij%7D)%5E2%7D%7B%5Csum_%7Bi%3Cj%7Dd_%7Bij%7D%5E2%7D%7D) | 고차원 거리와 저차원 거리 간의 차이 비율. 작을수록 거리 보존이 잘됨. |
| **[5-2] 거리 보존 - Sammon Error** | ![](https://latex.codecogs.com/svg.image?E_%7BSammon%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csum_%7Bi%3Cj%7D%20d_%7Bij%7D%7D%20%5Csum_%7Bi%3Cj%7D%20%5Cfrac%7B(d_%7Bij%7D-%5Chat%7Bd%7D_%7Bij%7D)%5E2%7D%7Bd_%7Bij%7D%7D) | 근접 관계를 강조한 거리 보존 오차. |
| **[6] 지역/전역 구조(LCMC)** | ![](https://latex.codecogs.com/svg.image?LCMC%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CN_H(i)%5Ccap%20N_L(i)%7C%7D%7Bk%7D%20-%20%5Cfrac%7Bk%7D%7Bn-1%7D) | 고차원/저차원 k-이웃의 겹침 비율로 지역/전역 구조를 함께 평가. |
| **[7] 쌍의 상관계수(Spearman’s ρ)** | ![](https://latex.codecogs.com/svg.image?%5Crho%20%3D%201%20-%20%5Cfrac%7B6%5Csum_%7Bi%3D1%7D%5E%7BN%7D(r_i%20-%20s_i)%5E2%7D%7BN(N%5E2%20-%201)%7D) | 거리 순위 일관성을 평가. ρ=1이면 완전히 동일한 순서. |
| **[8] Silhouette Score** | ![](https://latex.codecogs.com/svg.image?s(i)%20%3D%20%5Cfrac%7Bb(i)%20-%20a(i)%7D%7B%5Cmax(a(i)%2C%20b(i))%7D) | 군집 간 거리 대비 군집 내 밀집도 평가. |
| **[9] Davies–Bouldin Index(DBI)** | ![](https://latex.codecogs.com/svg.image?DBI%20%3D%20%5Cfrac%7B1%7D%7Bk%7D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Cmax_%7Bj%5Cne%20i%7D%20%5Cfrac%7B%5Csigma_i%2B%5Csigma_j%7D%7Bd(c_i%2Cc_j)%7D) | 군집 내 분산과 군집 간 중심 거리의 비율. 낮을수록 좋음. |
| **[10] Adjusted Rand Index(ARI)** | ![](https://latex.codecogs.com/svg.image?ARI%20%3D%20%5Cfrac%7B%5Csum_%7Bij%7D%20%5Cbinom%7Bn_%7Bij%7D%7D%7B2%7D%20-%20%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%2F%5Cbinom%7Bn%7D%7B2%7D%7D%7B%5Cfrac%7B1%7D%7B2%7D%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%20%2B%20%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%20-%20%5B%5Csum_i%20%5Cbinom%7Ba_i%7D%7B2%7D%5Csum_j%20%5Cbinom%7Bb_j%7D%7B2%7D%5D%2F%5Cbinom%7Bn%7D%7B2%7D%7D) | 군집 일치도 평가. 1이면 완벽 일치, 0은 무작위 수준. |
| **[11] Normalized Mutual Information(NMI)** | ![](https://latex.codecogs.com/svg.image?NMI(U%2CV)%20%3D%20%5Cfrac%7B2I(U%3BV)%7D%7BH(U)%20%2B%20H(V)%7D) | 군집 결과와 실제 레이블 간의 상호 정보량을 정규화. 값이 1에 가까울수록 유사도가 높음. |


<br>


## ▣ 차원축소 평가지표 결과해석

| 지표명                                   | 목표      | 권장 해석 기준                                       | 비고                                                      |
| ------------------------------------- | ------- | --------------------------------------------------- | ------------------------------------------------------- |
| **[1] 재구성 오류(Reconstruction Error)**  | `↓` | **≤ 0.05 우수**, **0.05~0.10 양호**, **> 0.10 미흡** | (정규화 MSE) 주로 선형 DR(PCA)·오토인코더에서 사용. k 스윕 후 엘보우로 차원 결정 권장  |
| **[2] 분산 유지율(Explained Variance Ratio)** | ↑ | **≥ 0.95 우수**, **0.90~0.95 양호**, **< 0.90 미흡** | (누적 EVR) PCA 등 선형 DR에 특화. 도메인상 필요한 분산 비율을 사전에 정하면 좋음  |
| **[3] 상호 정보량(Mutual Information)** | ↑ | **0.6 이상 양호** | (NMI 권장값) 절대 임계치보다는 동일 조건 간 상대 비교. 라벨있을 때 원라벨 vs 임베딩 기반 군집/축의 정보량. 규격화(NMI)가 해석에 유리|
| **[4-1] 근접도 보존 – Trustworthiness**  | ↑ | **≥ 0.95 우수**, **0.90~0.95 양호**, **0.85~0.89 보통**, **< 0.85 미흡** | 저차원 이웃 중 거짓 이웃이 얼마나 적은지. t-SNE, UMAP 평가에 흔용   |
| **[4-2] 근접도 보존 – Continuity** | ↑ | **≥ 0.95 우수**, **0.90~0.95 양호**, **0.85~0.89 보통**, **< 0.85 미흡** | 고차원 이웃이 저차원에서도 유지되는 정도. Trustworthiness와 쌍둥이 지표|
| **[5-1] 거리 보존 – Stress**  | `↓` | < **0.05 우수**, **0.05~0.10 양호**, **0.10~0.20 보통**, **> 0.20 미흡** | Kruskal stress. 고전 MDS 기준. 거리 스케일에 민감하므로 표준화 권장 |
| **[5-2] 거리 보존 – Sammon Error**  | `↓` | **0.1 내외 양호**  | 절대 임계치보단 상대 비교 및 엘보우 권장. 짧은 거리 가중이 큼. 지역 구조 보존에 민감. outlier 영향에 주의  |
| **[6] 지역/전역 구조(LCMC)** | ↑ | **≥ 0.90 우수**, **0.80~0.89 양호**, **0.70~0.79 보통**  | Local Continuity Meta Criterion. k-NN 일치율 기반 종합 지표 |
| **[7] 쌍의 상관계수(Spearman’s ρ)**  | ↑ | **≥ 0.90 우수**, **0.80~0.89 양호**, **0.60~0.79 보통**, **< 0.60 미흡** | 고차원 vs 저차원 쌍거리 순위 상관. 전역 기하 보존 평가. 계산량 큼(n²) |
| **[8] Silhouette Score**      | ↑ | **≥ 0.70 우수**, **0.50~0.69 양호**, **0.25~0.49 보통**, **< 0.25 미흡** | 임베딩에서 군집형성 품질 평가. 거리척도·스케일 영향.                      |
| **[9] Davies-Bouldin Index(DBI)**  | `↓` | **≤ 0.50 우수**, **0.51~0.99 양호**, **1.00~1.49 보통**, **≥ 1.50 미흡** | 군집 응집 대비 분리. 임베딩 스케일과 거리척도에 민감.                   |
| **[10] Adjusted Rand Index(ARI)**   | ↑ | **≥ 0.80 우수**, **0.60~0.79 양호**, **0.40~0.59 보통**, **< 0.40 미흡** | 라벨이 있을 때 임베딩→군집 vs 정답라벨 합치도. 범위 −1~1.               |
| **[11] Normalized Mutual Information(NMI)** | ↑ | **≥ 0.70 우수**, **0.50~0.69 양호**, **0.30–0.49 보통**, **< 0.30 미흡** | 라벨이 있을 때 규격화된 정보량. 0~1. 군집 수 차이에 덜 민감.     |



### Iris 데이터 + PCA(Principal Component Analysis) 학습 후 평가지표 11종 출력 소스

**[1] 재구성 오류(Reconstruction Error)**
<br>
**[2] 분산 유지율(Explained Variance Ratio)**
<br>
**[3] 상호 정보량(Mutual Information)**
<br>
**[4-1] 근접도 보존(Trustworthiness, Continuity)**
<br>
**[4-2] 근접도 보존(Trustworthiness, Continuity)**
<br>
**[5-1] 거리/유사도 보존(Stress, Sammon Error)**
<br>
**[5-2] 거리/유사도 보존(Stress, Sammon Error)**
<br>
**[6] 지역/전역구조(Local Continuity Meta Criterion)**
<br>
**[7] 쌍의 상관계수(Spearman’s ρ)**
<br>
**[8] Silhouette Score**
<br>
**[9] Davies-Bouldin Index(DBI)**
<br>
**[10] Adjusted Rand Index(ARI)**
<br>
**[11] Normalized Mutual Information(NMI)**
<br>



	# ============================================================
	# Iris + PCA(2D) 차원축소 후 11가지 평가지표 계산 (Windows 경고 억제 포함)
	# Requirements:
	#   pip install numpy pandas scikit-learn
	# ============================================================
	
	# ==== Windows 경고/스레드 이슈 해결용 (맨 위에 두세요) ====
	import os, warnings
	
	# (1) loky 물리코어 탐지 경고 억제: 논리 코어 수를 상한으로 지정
	os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
	
	# (2) MKL/OpenMP 스레드 수 제한: KMeans 메모리릭 경고 방지
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	os.environ.setdefault("MKL_NUM_THREADS", "1")
	os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
	os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
	os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
	
	# (선택) 관련 경고 메시지 숨김
	warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
	warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
	# ============================================================
	
	import numpy as np
	import pandas as pd
	from sklearn import datasets
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import (
	    silhouette_score, davies_bouldin_score, adjusted_rand_score,
	    normalized_mutual_info_score, pairwise_distances
	)
	from sklearn.cluster import KMeans
	from sklearn.feature_selection import mutual_info_classif
	
	
	# -----------------------------
	# Utilities (no-scipy required)
	# -----------------------------
	def _pairwise_orders(X):
	    """Return argsort indices (ascending distances) for each row, including self at position 0."""
	    D = pairwise_distances(X, metric="euclidean")
	    order = np.argsort(D, axis=1)
	    return D, order
	
	def _inverse_order(order):
	    """inv[i, j] = position of j in 'order[i]' (0=self, 1=nearest, ...)"""
	    n = order.shape[0]
	    inv = np.empty_like(order)
	    for i in range(n):
	        inv[i, order[i]] = np.arange(n)
	    return inv
	
	def trustworthiness_custom(X_high, X_low, n_neighbors=10):
	    """
	    Trustworthiness (Venna & Kaski, scikit-learn과 동일 정의).
	    저차원에서 k-이웃이지만 고차원에선 멀리 있는 '침입자' 페널티 합산.
	    """
	    n = X_high.shape[0]
	    _, order_high = _pairwise_orders(X_high)
	    _, order_low  = _pairwise_orders(X_low)
	    inv_high = _inverse_order(order_high)
	
	    T = 0.0
	    for i in range(n):
	        Nl = set(order_low[i][1:n_neighbors+1])  # low-d k-NN
	        Nh = set(order_high[i][1:n_neighbors+1]) # high-d k-NN
	        V  = Nl - Nh  # intrusions
	        for j in V:
	            r_ij_high = int(inv_high[i, j])  # 1..n-1 (0=self)
	            T += (r_ij_high - n_neighbors)
	
	    denom = n * n_neighbors * (2*n - 3*n_neighbors - 1)
	    if denom <= 0:
	        return np.nan
	    return 1.0 - (2.0 / denom) * T
	
	def continuity_custom(X_high, X_low, n_neighbors=10):
	    """
	    Continuity: trustworthiness의 반대 방향(고차원 이웃이 저차원에서 누락되면 페널티).
	    """
	    n = X_high.shape[0]
	    _, order_high = _pairwise_orders(X_high)
	    _, order_low  = _pairwise_orders(X_low)
	    inv_low = _inverse_order(order_low)
	
	    C = 0.0
	    for i in range(n):
	        Nh = set(order_high[i][1:n_neighbors+1]) # high-d k-NN
	        Nl = set(order_low[i][1:n_neighbors+1])  # low-d k-NN
	        U  = Nh - Nl  # omissions
	        for j in U:
	            r_ij_low = int(inv_low[i, j])  # 1..n-1 (0=self)
	            C += (r_ij_low - n_neighbors)
	
	    denom = n * n_neighbors * (2*n - 3*n_neighbors - 1)
	    if denom <= 0:
	        return np.nan
	    return 1.0 - (2.0 / denom) * C
	
	def kruskal_stress(X_high, X_low, eps=1e-12):
	    """Kruskal's Stress-1: sqrt( sum (dY - dX)^2 / sum dX^2 ). Lower is better."""
	    dX = pairwise_distances(X_high, metric="euclidean")
	    dY = pairwise_distances(X_low,  metric="euclidean")
	    iu = np.triu_indices_from(dX, k=1)
	    num = np.sum((dY[iu] - dX[iu])**2)
	    den = np.sum(dX[iu]**2) + eps
	    return float(np.sqrt(num / den))
	
	def sammon_error(X_high, X_low, eps=1e-12):
	    """
	    Sammon’s stress (Sammon mapping error):
	      E = (1 / sum dX) * sum ((dY - dX)^2 / dX), over i<j
	    Lower is better.
	    """
	    dX = pairwise_distances(X_high, metric="euclidean")
	    dY = pairwise_distances(X_low,  metric="euclidean")
	    iu = np.triu_indices_from(dX, k=1)
	    dX_u = dX[iu]
	    dY_u = dY[iu]
	    mask = dX_u > eps
	    num = np.sum(((dY_u[mask] - dX_u[mask])**2) / dX_u[mask])
	    den = np.sum(dX_u[mask]) + eps
	    return float(num / den)
	
	def lcmc(X_high, X_low, n_neighbors=10):
	    """
	    Local Continuity Meta-Criterion at fixed k:
	      Q_NX(k) = (1/(n*k)) * sum_i |N_high(i,k) ∩ N_low(i,k)|
	      LCMC(k)  = Q_NX(k) - k/(n-1)
	    """
	    n = X_high.shape[0]
	    _, order_high = _pairwise_orders(X_high)
	    _, order_low  = _pairwise_orders(X_low)
	
	    inter_total = 0
	    for i in range(n):
	        Nh = set(order_high[i][1:n_neighbors+1])
	        Nl = set(order_low[i][1:n_neighbors+1])
	        inter_total += len(Nh & Nl)
	    Q_NX = inter_total / (n * n_neighbors)
	    LCMC_k = Q_NX - n_neighbors / (n - 1)
	    return float(LCMC_k), float(Q_NX)
	
	def spearman_r_from_distances(X_high, X_low):
	    """
	    SciPy 없이 Spearman rho 계산(순위→피어슨).
	    """
	    dX = pairwise_distances(X_high, metric="euclidean")
	    dY = pairwise_distances(X_low,  metric="euclidean")
	    iu = np.triu_indices_from(dX, k=1)
	    a = dX[iu]
	    b = dY[iu]
	
	    # 동순위 평균을 고려한 간단 랭크 변환
	    def rank_avg(v):
	        order = np.argsort(v)
	        ranks = np.empty_like(order, dtype=float)
	        ranks[order] = np.arange(len(v), dtype=float)
	        # 동순위 처리
	        buckets = {}
	        for idx, val in enumerate(v):
	            buckets.setdefault(val, []).append(idx)
	        for idxs in buckets.values():
	            if len(idxs) > 1:
	                avg = float(np.mean(ranks[idxs]))
	                ranks[idxs] = avg
	        return ranks
	
	    ra = rank_avg(a)
	    rb = rank_avg(b)
	    # 피어슨 상관
	    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
	    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
	    return float(np.dot(ra, rb) / len(ra))
	
	
	# -----------------------------
	# Main
	# -----------------------------
	if __name__ == "__main__":
	    # Parameters (원하시면 변경)
	    n_components = 2
	    k_neighbors  = 10
	
	    # Data
	    iris = datasets.load_iris()
	    X = iris.data
	    y = iris.target
	
	    # Standardize
	    scaler = StandardScaler()
	    X_std = scaler.fit_transform(X)
	
	    # PCA (2D)
	    pca = PCA(n_components=n_components, random_state=42)
	    Z = pca.fit_transform(X_std)
	
	    # Reconstruction in standardized space
	    X_rec = pca.inverse_transform(Z)
	
	    # ---- Metrics ----
	    rec_err = float(np.mean((X_std - X_rec)**2))                         # [1] Reconstruction Error
	    evr_sum = float(np.sum(pca.explained_variance_ratio_))               # [2] EVR sum
	    mi_avg  = float(np.mean(mutual_info_classif(Z, y, random_state=42))) # [3] MI avg (components vs y)
	    trust   = float(trustworthiness_custom(X_std, Z, n_neighbors=k_neighbors)) # [4-1] Trustworthiness
	    cont    = float(continuity_custom(X_std, Z, n_neighbors=k_neighbors))      # [4-2] Continuity
	    stress  = float(kruskal_stress(X_std, Z))                            # [5-1] Kruskal Stress-1 (↓)
	    sammon  = float(sammon_error(X_std, Z))                              # [5-2] Sammon Error (↓)
	    lcmc_k, qnx = lcmc(X_std, Z, n_neighbors=k_neighbors)                # [6]  LCMC@k (with Q_NX)
	    rho     = float(spearman_r_from_distances(X_std, Z))                 # [7]  Spearman ρ
	
	    # Clustering-based metrics (on embedding)
	    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
	    labels_pred = kmeans.fit_predict(Z)
	
	    sil = float(silhouette_score(Z, y, metric="euclidean"))              # [8] Silhouette (y on Z)
	    dbi = float(davies_bouldin_score(Z, y))                              # [9] DBI (↓)
	    ari = float(adjusted_rand_score(y, labels_pred))                     # [10] ARI
	    nmi = float(normalized_mutual_info_score(y, labels_pred))            # [11] NMI
	
	    rows = [
	        ("[1]  Reconstruction Error (MSE, std space)", rec_err),
	        ("[2]  Explained Variance Ratio (sum, 2 comps)", evr_sum),
	        ("[3]  Mutual Information avg(Z_i; y)", mi_avg),
	        (f"[4-1] Trustworthiness@k={k_neighbors}", trust),
	        (f"[4-2] Continuity@k={k_neighbors}", cont),
	        ("[5-1] Kruskal Stress-1 (↓)", stress),
	        ("[5-2] Sammon Error (↓)", sammon),
	        (f"[6]  LCMC@k={k_neighbors}", lcmc_k),
	        ("[7]  Spearman ρ (pairwise distances)", rho),
	        ("[8]  Silhouette Score (using y on Z)", sil),
	        ("[9]  Davies–Bouldin Index (using y on Z, ↓)", dbi),
	        ("[10] Adjusted Rand Index (KMeans(Z) vs y)", ari),
	        ("[11] Normalized Mutual Information (KMeans(Z) vs y)", nmi),
	    ]
	    df = pd.DataFrame(rows, columns=["Metric", "Value"])
	
	    print("\nIris PCA (2D) Evaluation Summary\n" + "-"*42)
	    print(df.to_string(index=False))
	
	    print("\nExplained Variance Ratio per component:")
	    for i, r in enumerate(pca.explained_variance_ratio_, start=1):
	        print(f"  PC{i}: {r:.6f}")



### (소스 실행 결과)

	Iris PCA (2D) Evaluation Summary
	------------------------------------------
	Metric    Value
	[1]  Reconstruction Error (MSE, std space) 0.041868
	[2]  Explained Variance Ratio (sum, 2 comps) 0.958132
	[3]  Mutual Information avg(Z_i; y) 0.551648
	[4-1] Trustworthiness@k=10 0.977963
	[4-2] Continuity@k=10 0.990622
	[5-1] Kruskal Stress-1 (↓) 0.062736
	[5-2] Sammon Error (↓) 0.009755
	[6]  LCMC@k=10 0.665553
	[7]  Spearman ρ (pairwise distances) 0.993385
	[8]  Silhouette Score (using y on Z) 0.401387
	[9]  Davies–Bouldin Index (using y on Z, ↓) 0.955460
	[10] Adjusted Rand Index (KMeans(Z) vs y) 0.620135
	[11] Normalized Mutual Information (KMeans(Z) vs y) 0.659487

	Explained Variance Ratio per component:
  	PC1: 0.729624
  	PC2: 0.228508

  
### (결과 분석)

	[1]  Reconstruction Error (MSE, std space) 0.041868 → 우수
	[2]  Explained Variance Ratio (sum, 2 comps) 0.958132 → 우수
	[3]  Mutual Information avg(Z_i; y) 0.551648 → 양호
	[4-1] Trustworthiness@k=10 0.977963 → 우수
	[4-2] Continuity@k=10 0.990622 → 우수
	[5-1] Kruskal Stress-1 (↓) 0.062736 → 우수
	[5-2] Sammon Error (↓) 0.009755 → 우수
	[6]  LCMC@k=10 0.665553 → 우수
	[7]  Spearman ρ (pairwise distances) 0.993385 → 우수
	[8]  Silhouette Score (using y on Z) 0.401387 → 양호
	[9]  Davies–Bouldin Index (using y on Z, ↓) 0.955460 → 양호
	[10] Adjusted Rand Index (KMeans(Z) vs y) 0.620135 → 양호
	[11] Normalized Mutual Information (KMeans(Z) vs y) 0.659487 → 양호


<!--

## [차원 축소 알고리즘 평가 사용방법]

**▣ 재구성 오류(Reconstruction Error) :** 차원 축소된 데이터를 원본 차원으로 복원하여 복원된 데이터와 원본 데이터 간의 평균 제곱 오차(MSE)를 통해 재구성 오류를 계산<br>

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)를 준비
    data = load_iris()
    X = data.data  # 입력 데이터 (특성)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)  # 차원 축소된 데이터

    # 재구성 오류 계산 : 차원 축소된 데이터를 원래 차원으로 복원하고 원본 데이터와의 평균 제곱 오차(MSE)를 계산
    X_reconstructed = pca.inverse_transform(X_reduced)  # 차원 축소 후 복원된 데이터
    reconstruction_error = mean_squared_error(X, X_reconstructed)  # 재구성 오류 계산
    print(f"Reconstruction Error (MSE): {reconstruction_error:.3f}")

<br>

**▣ 분산 유지율(Explained Variance Ratio) :** 각 주성분이 설명하는 분산 비율을 통해 데이터의 정보 손실 정도를 파악

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)를 준비합니다.
    data = load_iris()
    X = data.data  # 입력 데이터 (특성)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    pca.fit(X)  # PCA 학습

    # 분산 유지율 계산 : 각 주성분이 데이터의 분산을 얼마나 설명하는지 비율로 확인
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio per Component: {explained_variance_ratio}")
    print(f"Total Variance Retained: {sum(explained_variance_ratio):.3f}")  # 전체 분산 유지율

<br>

**▣ 상호 정보량(Mutual Information) :** 차원 축소 전후 데이터의 정보량을 비교

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    # 데이터 로드 : Iris 데이터셋을 로드하여 입력 데이터(X)와 실제 레이블(y_true)를 준비
    data = load_iris()
    X = data.data         # 입력 데이터 (특성)
    y_true = data.target  # 실제 레이블 (클러스터링 평가 시 사용)

    # PCA를 사용하여 주성분 개수를 2개로 설정하여 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)  # 차원 축소된 데이터

    # KMeans를 사용하여 차원 축소된 데이터에서 클러스터링을 수행 : 클러스터 개수를 3으로 설정하여 실제 클래스 수와 맞추기
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X_reduced)  # 클러스터링 예측 레이블

    # 4. 상호 정보량 계산
    # (1) Adjusted Mutual Information (AMI) : 실제 레이블(y_true)과 클러스터링 예측 레이블(y_pred) 간의 유사도를 측정
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print(f"Adjusted Mutual Information (AMI): {ami:.3f}")

    # (2) Normalized Mutual Information (NMI) : 실제 레이블과 예측 레이블 간의 상호 정보량을 정규화하여 측정
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

<br>

**▣ 군집 평가 지표 :** 차원 축소 후 클러스터링을 수행하고 군집 평가 지표를 계산하여 차원 축소의 성능을 평가

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.model_selection import train_test_split

    # 데이터 로드
    data = load_iris()
    X = data.data
    y_true = data.target  # 실제 레이블 (평가를 위해 사용)

    # PCA를 사용하여 차원 축소
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # KMeans를 사용하여 클러스터링 수행
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X_reduced)

    # 군집 평가 지표 계산
    # (1) Silhouette Score
    silhouette = silhouette_score(X_reduced, y_pred)
    print(f"Silhouette Score: {silhouette:.3f}")

    # (2) Davies-Bouldin Index (DBI) - 클러스터들이 얼마나 잘 분리되고 응집되어 있는지 평가(DBI가 낮을수록 클러스터링 품질이 더 좋음)
    davies_bouldin = davies_bouldin_score(X_reduced, y_pred)
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

    # (3) Adjusted Rand Index (ARI) - 실제 레이블과 예측 레이블 비교(클러스터링 결과와 실제 레이블 간의 일치도를 측정: 1에 가까울 수록 유사)
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    # (4) Normalized Mutual Information (NMI) - 실제 레이블과 예측 레이블 비교(클러스터링 결과와 실제 레이블 간의 정보량의 공유 정도를 측정: 1에 가까울 수록 유사)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

<br>
-->
                        

