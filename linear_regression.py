import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# قراءة مجموعة البيانات
dataset = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\archive (1)\data.csv")

# تحديد المتغيرات المستقلة والتابعة
X = dataset.drop(labels=['Bankrupt?'], axis=1)
y = dataset['Bankrupt?']

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# إنشاء وتدريب نموذج الانحدار اللوجستي
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# استخدام النموذج للتنبؤ بقيمة التصنيف لبيانات الاختبار
y_pred = log_reg.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# تنبؤ القيم باستخدام النموذج
y_pred = log_reg.predict(X_test)

# حساب مصفوفة الارتباط
conf_matrix = confusion_matrix(y_test, y_pred)

# عرض مصفوفة الارتباط
print("Confusion Matrix:")
print(conf_matrix)

# مخطط ROC
y_pred_proba = log_reg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# عرض مخطط ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
