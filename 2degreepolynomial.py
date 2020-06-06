
# please install numpy and plt as packages




# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculating mean_squared error
from sklearn.metrics import mean_squared_error

# creating a dataset with curvilinear relationship
x=33.28*np.random.normal(0,1,70)
y=33.28+np.random.normal(-100,100,70)

# plotting dataset
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()
# Importing Linear Regression



from sklearn.linear_model import LinearRegression

# Training Model for the linear regression model
lm=LinearRegression()
lm.fit(x.reshape(-1,1),y.reshape(-1,1))
y_pred=lm.predict(x.reshape(-1,1))

# plotting predictions
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r')
plt.xlabel(' This is the x',fontsize=16)
plt.ylabel('This is line for mean average',fontsize=16)
plt.show()



