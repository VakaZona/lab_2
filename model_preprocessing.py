import pandas as pd

df_train = pd.read_csv('/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/train/train.csv')
df_test = pd.read_csv('/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/test/test.csv')

def percentConvert(percent):
    if (percent > 30):
        return 1
    else:
        return 0

def saveData(df, path):
    df.to_csv(path, index=False)

def year(date):
    return date.split('-')[0]
def month(date):
    return date.split('-')[1]
def day(date):
    return date.split('-')[2]
def hour(time):
     return time.split(':')[0]
def minute(time):
     return time.split(':')[1]

def temperatureConvert(temperature):
    return round(temperature, 0)

df_train['Year'] = df_train['Date'].apply(lambda x: year(x))
df_train['Month'] = df_train['Date'].apply(lambda x: month(x))
df_train['Day'] = df_train['Date'].apply(lambda x: day(x))
df_train['Hour'] = df_train['Time'].apply(lambda x: hour(x))
df_train['Minute'] = df_train['Time'].apply(lambda x: minute(x))
df_train['Precipitation'] = df_train['Precipitation_procent'].apply(lambda x: percentConvert(x))
df_train['Temperature'] = df_train['Temperature'].apply(lambda x: temperatureConvert(x))
df_train=df_train.drop(['Precipitation_procent', 'Date', 'Time'], axis=1)
df_train[['Year','Month', 'Day', 'Temperature', 'Precipitation', 'Hour', 'Minute']]=df_train[['Year','Month', 'Day', 'Temperature', 'Precipitation','Hour', 'Minute']].astype(int)

df_test['Year'] = df_test['Date'].apply(lambda x: year(x))
df_test['Month'] = df_test['Date'].apply(lambda x: month(x))
df_test['Day'] = df_test['Date'].apply(lambda x: day(x))
df_test['Hour'] = df_test['Time'].apply(lambda x: hour(x))
df_test['Minute'] = df_test['Time'].apply(lambda x: minute(x))
df_test['Precipitation'] = df_test['Precipitation_procent'].apply(lambda x: percentConvert(x))
df_test['Temperature'] = df_test['Temperature'].apply(lambda x: temperatureConvert(x))
df_test=df_test.drop(['Precipitation_procent', 'Date', 'Time'], axis=1)
df_test[['Year','Month', 'Day', 'Temperature', 'Precipitation', 'Hour', 'Minute']]=df_test[['Year','Month', 'Day', 'Temperature', 'Precipitation','Hour', 'Minute']].astype(int)



saveData(df_train, '/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/train/train_preprocessing.csv')
saveData(df_test, '/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/test/test_preprocessing.csv')
