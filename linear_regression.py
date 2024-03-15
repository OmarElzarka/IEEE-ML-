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
