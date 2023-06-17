import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


df = pd.read_csv('/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/train/train_preprocessing.csv')
# df.info()

X = df.drop('Precipitation', axis=1)
y = df.Precipitation

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
xgr = XGBClassifier(random_state=30)


xgr = xgr.fit(x_train, y_train)

test_predictions = xgr.predict(x_test)

mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R-squared: {r2:.3f}")

filename = 'final_model.sav'
pickle.dump(xgr, open(filename, 'wb'))