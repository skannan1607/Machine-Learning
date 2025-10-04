
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

week_data = [1,2,3,4,5]
sale_data = [2,4,5,4,5]
df = pd.DataFrame({"Weeks":week_data,"Sales":sale_data})
df

X = df['Weeks'].values
Y = df['Sales'].values
mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)
num,den=0,0
for i in range(n):
  num+= (X[i]-mean_x)*(Y[i]-mean_y)
  den+= (X[i]-mean_x)**2
m = num/den
c = mean_y-(m*mean_x)
print(f"m:{m}")
print(f"c:{c}")

max_x=np.max(X)+1
min_X=np.min(X)-1

x=np.linspace(min_X,max_x)
y=c+m*x
print(f"max_x:{max_x}")
print(f"min_x:{min_X}")
plt.plot(x,y,color='Black',label='Regression Line',linestyle='--')
plt.scatter(X,Y,c='#ef5423',label='Scatter Plot')
plt.xlabel('Weeks')
plt.ylabel('Sales')
plt.legend(loc='best')
plt.grid()
plt.show()

#root mean square
ss_tot = 0
ss_res = 0

for i in range(n):
  y_pred = (m * X[i]) + c
  ss_tot += (Y[i] - mean_y) ** 2
  ss_res += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_res / ss_tot)
print(f"r2 :{r2}")

#validate the model
from sklearn.linear_model import LinearRegression

X = X.reshape((n,1))
print(X)
reg = LinearRegression()
reg = reg.fit(X,Y)

R2 = reg.score(X,Y)
print(R2)
