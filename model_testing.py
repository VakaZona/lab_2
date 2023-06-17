import pandas as pd
import pickle

df = pd.read_csv('/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/test/test_preprocessing.csv')

X = df.drop('Precipitation', axis=1)
y = df.Precipitation

filename= 'final_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X,y)
print(result)
