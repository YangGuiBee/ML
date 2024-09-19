#  14 : 분류 평가 지표

---
	
 	[1] 오차행렬, 혼동행렬 (Confusion Matrix)
  	[2] 정확도 (Accurancy)
	[3] 정밀도 (Precision), PPV (Positive Predictive Value)
	[4] 재현율 (Recall), 민감도 (Sensitivity), TPR (True Positive Rate)
	[5] F1 score
	[6] ROC curve
	[7] AUC score
	  
---

# [1] 오차행렬, 혼동행렬 (Confusion Matrix)
분류 모델의 평가 기준<br>
![](./images/CM_table.PNG)
<br>
TP(True Positive): 모델이 positive라고 예측했는데 실제로 정답이 positive (정답)<br>
TN(True Negative): 모델이 negative라고 예측했는데 실제로 정답이 negative (정답)<br>
FP(False Positive): 모델이 positive라고 예측했는데 실제로 정답이 negative (오답)<br>
FN(False Negative): 모델이 negative라고 예측했는데 실제로 정답이 positive (오답)<br>

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
$Precision = \frac{TP}{TP + FP}$<br>
<br>
실제로 정답이 positive인 것들 중에서 모델이 positive라고 예측한 비율<br>
0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋다.<br>
실제 정답이 positive인 데이터를 negative라고 잘못 예측하면 안 되는 경우에 중요한 지표가 될 수 있다.<br>
Recall를 높이기 위해선 FN(모델이 negative라고 예측했는데 정답이 positive인 경우)을 낮추는 것이 중요하다.<br>

<br>

# [5] F1 score

<br>

# [6] ROC curve

<br>

# [7] AUC score

<br>
