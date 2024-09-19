#  14 : 분류 평가 지표

---
	
 	[1] 오차행렬, 혼동행렬 (Confusion Matrix)
  	[2] 정확도 (Accurancy)
	[3] 정밀도 (Precision), PPV (Positive Predictive Value)
	[4] 재현율 (Recall), 민감도 (Sensitivity), TPR (True Positive Rate)
	[5] F1 score
 	[6] 오분류율 (Error Rate)
  	[7] 특이도 (Specificity), TNR(True Negative Rate)
   	[8] 위양성률 (Fall Out), FPR(False Positive Rate)
	[9] ROC curve
	[10]AUC score
	  
---

# [1] 오차행렬, 혼동행렬 (Confusion Matrix)
분류 모델의 평가 기준<br>
![](./images/CM_table.PNG)
<br>
TP(True Positive): 모델이 positive라고 예측했는데 실제로 정답이 positive (정답)<br>
TN(True Negative): 모델이 negative라고 예측했는데 실제로 정답이 negative (정답)<br>
FP(False Positive): 모델이 positive라고 예측했는데 실제로 정답이 negative (오답)<br>
FN(False Negative): 모델이 negative라고 예측했는데 실제로 정답이 positive (오답)<br>
![](./images/CM_table_real.PNG)

<br>

# [1] 정확도 (Accurancy)
$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$<br>
<br>
모델이 전체 문제 중에서 정답을 맞춘 비율<br>
0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋다.<br>
데이터가 불균형할 때(positive:negative=9:1)는 Accuracy만으로 제대로 분류했는지는 알 수 없기 때문에 Recall과 Precision을 사용<br>

<br>

# [3] 정밀도 (Precision), PPV(Positive Predictive Value)
$Precision = \frac{TP}{TP + FP}$<br>
<br>
모델이 positive라고 예측한 것들 중에서 실제로 정답이 positive인 비율<br>
0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋다.<br>
실제 정답이 negative인 데이터를 positive라고 잘못 예측하면 안 되는 경우에 중요한 지표가 될 수 있다.<br>
Precision을 높이기 위해선 FP(모델이 positive라고 예측했는데 정답은 negative인 경우)를 낮추는 것이 중요하다.<br>

<br>

# [4] 재현율 (Recall), 민감도 (Sensitivity), TPR (True Positive Rate)
$Recall = \frac{TP}{TP + FN}$<br>
<br>
실제로 정답이 positive인 것들 중에서 모델이 positive라고 예측한 비율<br>
0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋다.<br>
실제 정답이 positive인 데이터를 negative라고 잘못 예측하면 안 되는 경우에 중요한 지표가 될 수 있다.<br>
Recall를 높이기 위해선 FN(모델이 negative라고 예측했는데 정답이 positive인 경우)을 낮추는 것이 중요하다.<br>

<br>

# [5] F1 score
$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$<br>
<br>
Recall과 Precision의 조화평균<br>
0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋다.<br>
Recall과 Precision은 상호 보완적인 평가 지표이기 때문에 F1 score를 사용한다.<br>
Precision과 Recall이 한쪽으로 치우쳐지지 않고 모두 클 때 큰 값을 가진다.<br>

<br>

# [6] 오분류율 (Error Rate)
$Accuracy = \frac{FP + FN}{TP + TN + FP + FN}$<br>
<br>
모델이 전체 데이터에서 잘못 맞춘 비율<br>

<br>

# [7] 특이도 (Specificity), TNR(True Negative Rate)
$Specificity = \frac{TN}{TN + FP}$<br>
<br>
실제 정답이 negative인 것들 중에서 모델이 negative라고 예측한 비율<br>

<br>

# [8] 위양성률 (Fall Out), FPR(False Positive Rate)
$Fall Out = 1 - Specificity = 1 - \frac{TN}{TN + FP} = \frac{FP}{FP + TN}$<br>
<br>
실제 정답이 negative인 것들 중에서 모델이 positive라고 예측한 비율<br>
<br>

# [9] ROC curve
FPR을 X축, TPR을 Y축으로 놓고 임계값을 변경해서 FPR이 변할 때 TPR이 어떻게 변하는지 나타내는 곡선<br>
여러 임계값들을 기준으로 Recall-Fallout의 변화를 시각화한 것<br>
Fallout은 실제 False인 data 중에서 모델이 True로 분류항 비율을<br>
Recall은 실제 True인 data 중에서 모델이 True로 분류한 비율을 나타낸 지표로써,<br> 
이 두 지표를 각각 x, y의 축으로 놓고 그려지는 그래프를 해석<br>

<br>

# [10] AUC score
ROC 곡선 아래쪽 면적<br>
1.0 ~ 0.9 : 아주 좋음<br>
0.9 ~ 0.8 : 좋음<br>
0.8 ~ 0.7 : 괜찮은 모델<br>
0.7 ~ 0.6 : 의미는 있으나 좋은 모델은 아님<br>
0.6 ~ 0.5 : 좋지 않은 모델<br>

<br>
