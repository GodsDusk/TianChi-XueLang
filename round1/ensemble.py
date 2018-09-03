import pandas as pd 
import datetime

df1 = pd.read_csv('classical/submission.csv')

df2 = pd.read_csv('siamese/submission.csv')



assert all(df1.filename.values == df2.filename.values)


df1.probability = (df1.probability + df2.probability) / 2.

df1.to_csv('submit/submit_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)