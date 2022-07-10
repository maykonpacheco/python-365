import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())

# look at whether this dataset contains any null values or not.
print(data.isnull().sum())

import plotly.express as px
import plotly.graph_objects as go

# TV plotly
figure_tv = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure_tv.show()

# Newspaper plotly
figure_newspaper = px.scatter(data_frame= data, x="Sales",
                             y="Newspaper", size="Newspaper", trendline="ols")
figure_newspaper.show()

# Radio plotly
figure_radio = px.scatter(data_frame = data, x="Sales", 
                          y="Radio", size="Radio", trendline="ols"  
)
figure_radio.show()

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

# Spli the data
x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Train the model
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# Predict
#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))

