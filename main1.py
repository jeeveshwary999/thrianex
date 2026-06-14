import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data = {
    "Area":[1000,1500,2000,2500,3500,4000],
    "Bedroom":[2,3,3,4,4,5],
    "Price":[200000,300000,400000,500000,600000,700000]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)
x=df[['Area','Bedroom']]
y =df['Price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("\nActual Prices:")
print(y_test.values)
print("\nPredicted Prices:")
print(y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\nMean Squared Error:",mse)
print("R2 Score:",r2)
new_house = pd.DataFrame({
    "Area":[2000],
    "Bedroom":[3]
})
predicted_price = model.predict(new_house)
print("\nPredicted House Price for Area = 2200 and Bedrooms =3:")
print(predicted_price[0])
plt.figure(figsize=(6,4))
plt.scatter(y_test,y_pred,color='blue')
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color ='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

