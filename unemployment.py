import pandas as ps
import numpy as ny
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE

csv = "C:/Users/sasi1/OneDrive/Documents/abilash/internpedia/Unemployment in India.csv" 
dr = ps.read_csv(csv)

print(dr.head())
print(dr.isna().sum())
print(dr.describe())
dr.columns = dr.columns.str.strip()


dr['Date'] = ps.to_datetime(dr['Date'], dayfirst=True)
dr.dropna(inplace=True)
plt.figure(figsize=(10, 5))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=dr, marker='o')
plt.title('Estimated Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x='Region', y='Estimated Unemployment Rate (%)', data=dr)
plt.title('Estimated Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

corr = dr[['Estimated Unemployment Rate (%)', 'Estimated Labour Participation Rate (%)', 'Estimated Employed']].corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


X = dr[['Estimated Labour Participation Rate (%)', 'Estimated Employed']]  
y = dr['Estimated Unemployment Rate (%)'] 


Xtr, Xtst, ytr, ytst = tts(X, y, test_size=0.2, random_state=21)

proto_type = LR()
proto_type.fit(Xtr, ytr)

y_pre = proto_type.predict(Xtst)
mse = MSE(ytst, y_pre)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(12, 6))
plt.scatter(ytst, y_pre, color='blue', label='Actual vs Predicted')
plt.plot([ytst.min(), ytst.max()], [ytst.min(), ytst.max()], color='red', linewidth=2, label='Ideal Fit')
plt.title('Actual vs Predicted Unemployment Rate')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Predicted Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
