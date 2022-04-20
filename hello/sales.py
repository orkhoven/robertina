import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv(r'df_total.csv')
df.drop(columns = ["Unnamed: 0"], inplace = True)  

model = xgb.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)

every_column_except_y= [col for col in df.columns if col not in ['GL_NUMERO']]
model.fit(df[every_column_except_y].values,df['GL_NUMERO'].values)


pickle.dump(model, open('model.pkl', 'wb'))


