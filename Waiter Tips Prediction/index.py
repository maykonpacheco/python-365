import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("tips.csv")
print(data.head())


figure_week = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color="day", trendline="ols"
)
figure_week.show()


figure_gender = px.scatter(data_frame=data, x="total_bill",
                           y="tip", size="size", color="sex", trendline="ols" 
)
figure_week()


figure_time = px.scatter(data_frame=data, x="total_bill", 
                        y="tip", size="size", color="time", trendline="ols"
)
figure_time.show()


figure_pie_days = px.pie(data,
                values='tip',
                name='day', hole=0.5)
figure_pie_days.show()


figure_pie_gender = px.pie(data,
                          values='tip',
                          names='sex', hole=0.5)
figure_pie_gender.show()


figure_pie_smoker = px.pie(data,
                           values='tip',
                           names='smoker', hole=0.5)
figure_pie_smoker.show()                  


figure_pie_lunch_or_dinner = px.pie(data,
                           values='tip',
                           names='time', hole=0.5)
figure_pie_lunch_or_dinner.show()                  

# data transformation 
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()

# Split data
x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

# Prediction model
# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)