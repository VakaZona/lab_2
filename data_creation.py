from faker import Faker
import pandas as pd
import random


def generateData(n):
    df = pd.DataFrame(
        [
            {
                "Date": fake.date_this_century(),
                "Time": fake.time(),
                "Temperature": 0,
                "Precipitation_procent": fake.pyint(min_value=0, max_value=100, step=10)
            }
            for _ in range(n)
        ]
    )
    return df

def generateTemperature(date):
    if (date.month>=12 or date.month<=2):
        return round(random.uniform(-30,0),2)
    elif (date.month>=3 and date.month<=5):
        return round(random.uniform(0,25),2)
    elif (date.month>=6 and date.month<=8):
        return round(random.uniform(10,30),2)
    elif (date.month>=9 and date.month<=11):
        return round(random.uniform(-20,10),2)


def saveData(df, path):
    df.to_csv(path, index=False)

fake = Faker("ru_Ru")

df_train = generateData(1000)
df_train['Temperature'] = df_train['Date'].apply(lambda x: generateTemperature(x))
saveData(df_train, '/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/train/train.csv')

df_test = generateData(300)
df_test['Temperature'] = df_test['Date'].apply(lambda x: generateTemperature(x))
saveData(df_test, '/Users/valery/Учеба Агу/Автоматизация машинного обучения/lab_2/test/test.csv')
