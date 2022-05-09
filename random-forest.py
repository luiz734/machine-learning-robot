from sklearn.ensemble import RandomForestRegressor
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas

data = pandas.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
data.head()

x = data.drop(columns=["Saída Vel. Linear(m/s)","Saída Vel. Angular (rad/s.horário=positivo)"], axis=1)
y = data['Saída Vel. Linear(m/s)']

total = 0
total_sqrt = 0
for i in range(1000):
   print(i)
   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

   regressor = RandomForestRegressor(
      n_estimators=200,
      # max_features="sqrt",
      max_depth=15, 
      n_jobs=-1,
   )
   regressor.fit(X_train, y_train)
   result = regressor.predict(X_test).tolist()

   squared_error = mean_squared_error(y_test, result)
   total += squared_error;
   total_sqrt += sqrt(squared_error);

print(total / 1000)
print(total_sqrt / 1000)



