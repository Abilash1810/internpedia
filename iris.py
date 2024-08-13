import pandas as ps
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score as a_s, classification_report as c_r, confusion_matrix as c_m
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
csv = load_iris()
dr = ps.DataFrame(data=csv.data, columns=csv.feature_names)
dr['species'] = csv.target
print(dr.isna().sum())
dr['species'] = dr['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
sns.pairplot(dr, hue='species')
plt.show()
X_tr = dr.drop('species', axis=1)
Y_tr = dr['species']
Xtr, Xtst, ytr, ytst = tts(X_tr, Y_tr, test_size=0.2, random_state=21)
proto_type = RF(random_state=28)
proto_type.fit(Xtr, ytr)
y_pre = proto_type.predict(Xtst)
print("Accuracy:", a_s(ytst, y_pre))
print("\nClassification Report:\n", c_r(ytst, y_pre))
conf_mat = c_m(ytst, y_pre)
sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=csv.target_names, yticklabels=csv.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
