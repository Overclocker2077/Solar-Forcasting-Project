import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import random
from rich import print as pr

np.set_printoptions(precision=3, suppress=True)
sns.set_theme()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# print(tf.__version__)

# pre: 2d array of strings 
# post: returns a string array with the column names and a dictionary with the colomn name pair with the data in that column
def convertDataFrame(dataframe) -> dict[str, list[float]]:
    array = dataframe.to_numpy().tolist()
    column_names = array.pop(0)
    table = {}
    for name in column_names:
        table[name] = list()
    # print(table)
    for r in range(len(array)):
        for c in range(len(array[r])):
            table[column_names[c]].append(float(array[r][c]))
    return table

# pre: 2d array, and target_index is the values to be filter by the key_index, using the split_value
# post: returns two lists containing the divided data set based on the key_index and split_func
def split_filter(dict, target_index, key_index, split_func = lambda x : x > 65 and x < 80) -> tuple[list[float], list[float]]:
    array1 : list[float] = [] # room 
    array2 : list[float] = [] # below
    array3 : list[float] = [] # above
    for i in range(len(dict[target_index])):
        if (split_func(dict[key_index][i])):
            array1.append(dict[target_index][i])
        elif (dict[key_index][i] < 65):
            array2.append(dict[target_index][i])
        else:
            array3.append(dict[target_index][i])
    return array1, array2, array3

# 4th degree Avg loss: 0.11494852884846171
# 5th degree Avg loss: 0.058064411397904046
# 6th degree Avg loss: 0.46999121531433935

def Polynomial_regression(degree: int = 1) -> tuple[np.poly1d, np.poly1d, np.ndarray, np.ndarray, np.poly1d, np.ndarray]:
    fc_room, fc_below, fc_above = split_filter(dict = data_table, target_index=columns[2], key_index=columns[3])
    dcv_room, dcv_below, dcv_above = split_filter(dict = data_table, target_index=columns[4], key_index=columns[3])
    
    model_room_temp = np.poly1d(np.polyfit(x = fc_room,  y = dcv_room, deg = degree))
    polyline_room_temp = np.linspace(1, 4000, 50)

    model_above_temp = np.poly1d(np.polyfit(x = fc_above,  y = dcv_above, deg = degree))
    polyline_above_temp = np.linspace(1, 4000, 50)

    model_below_temp = np.poly1d(np.polyfit(x = fc_below,  y = dcv_below, deg = degree))
    polyline_below_temp = np.linspace(1, 4000, 50)

    return model_below_temp, model_room_temp, polyline_below_temp, polyline_room_temp, model_above_temp, polyline_above_temp

def plot_data():
    model_below_temp, model_room_temp, polyline_below_temp, polyline_room_temp, model_above_temp, polyline_above_temp = Polynomial_regression(4)
    
    #model_below_temp, model_room_temp, polyline_below_temp, polyline_room_temp = Logarithmic_regression()

    plt.scatter(x = data_table[columns[2]], y = data_table[columns[4]], label='Data', c = data_table[columns[3]], cmap="turbo")
    plt.plot(polyline_room_temp, model_room_temp(polyline_room_temp), color = "Orange")
    plt.plot(polyline_below_temp, model_below_temp(polyline_below_temp), color = "Blue")
    plt.plot(polyline_above_temp, model_above_temp(polyline_above_temp), color = "Red")
    print(model_below_temp)
    print(model_room_temp)
    print(model_above_temp)
    color_bar = plt.colorbar()
    color_bar.set_label("Temperature (F)")
    plt.xlabel('Light Intensity (Fc)')
    plt.ylabel('Energy Output (DCV)')
    plt.legend()
    plt.grid(True)
    #print(y_prediction)
    # print(y_prediction.shape)


class Polynomial_Prediction_Model():

    def __init__(self, df, predict_column, degree):
        self.df = df
        self.degree = degree
        self.columns = df.columns.tolist()
        self.columns.pop(4)
        self.predict_column = predict_column

        self.input_features = df[self.columns].values
        self.output_feature = df[self.predict_column].values
        self.model = None
        self.x_poly = None

    # Pre: Normaization axis automatically set to -1, epochs set to 100 
    # Post: train and returns training history 
    def compile(self, axis = -1):
        if (self.model == None):
            self.poly = PolynomialFeatures(degree = self.degree, include_bias = False)
            self.x_poly = self.poly.fit_transform(self.input_features)

            X_train, X_test, y_train, y_test = train_test_split(self.x_poly, self.output_feature, test_size=0.2, random_state=42)

            self.model = LinearRegression()
            self.model.fit(self.x_poly, self.output_feature)
        else:
            raise AttributeError("Model has already been compiled.")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        if (self.model != None):
            poly = PolynomialFeatures(degree = self.degree, include_bias = False)
            x_poly = poly.fit_transform(features)

            return self.model.predict(x_poly)
        raise AttributeError("Must compile the model before calling predict.")

    def calc_loss(self, output_prediction: np.ndarray) -> tuple[np.ndarray[float], float]:
        output_prediction : list = output_prediction.tolist()
        loss = []
        total = 0    
                                   #low     med    high
        avg_loss_values_sorted = [[0, 0], [0, 0], [0, 0]]  # categorize the data
        for i in range(len(output_prediction)):
            loss_val = abs(float(self.df["DCV Output"].iloc[i]) - float(output_prediction[i]))
            loss.append(loss_val)
            if (self.df["Fc"].iloc[i] < 750):
                avg_loss_values_sorted[0][0] += loss_val
                avg_loss_values_sorted[0][1] += 1
            elif (self.df["Fc"].iloc[i] < 1500):
                avg_loss_values_sorted[1][0] += loss_val
                avg_loss_values_sorted[1][1] += 1
            else:
                avg_loss_values_sorted[2][0] += loss_val
                avg_loss_values_sorted[2][1] += 1

            total += loss_val
        avg_loss = total / len(loss)
        avg_loss_low = avg_loss_values_sorted[0][0] / avg_loss_values_sorted[0][1]
        avg_loss_med = avg_loss_values_sorted[1][0] / avg_loss_values_sorted[1][1] 
        avg_loss_high = avg_loss_values_sorted[2][0] / avg_loss_values_sorted[2][1] 

        # loss.sort(key = lambda e : self.df["Fc"].iloc[e[1]])
        
        # for i in range(len(loss)):
        #     loss[i].pop()
    
        return [avg_loss_low, avg_loss_med , avg_loss_high], avg_loss
    
    def save(self, model_name):
        ...

    def load(self, path):
        ...

    def summary(self):
        ...

    def plot(self, x,y, name = "Data"):
        plt.scatter(self.df["Fc"], self.df[self.predict_column], label = name)
        plt.scatter(x, y, color="Orange", label = "Predictions")
        plt.ylim(0,6)
        plt.colorbar()
        plt.ylabel("Energy Output (DCV)")
        plt.xlabel("Light Intensity (Fc)")
        plt.legend()
    
    def plot_loss(self, loss: np.ndarray[float]):
        # x = [i for i in range(len(loss))]
        # # plt.scatter(x, loss, label='val_loss points')
        # plt.plot(loss, label='Loss Value')
        plt.bar(["Low (0 - 750)", "Medium (750 - 1500)", "High (> 1500)", "Total Avg Loss"], loss)
        plt.ylim([0, 0.3])
        plt.title("Average Loss Value: Categorized by light intensity")
        plt.xlabel('Light Intensity (Fc)')
        plt.ylabel(f'Abs. Prediction Error (DCV)')
        plt.legend()
        plt.grid(True)

def generate_weather_data(**args) -> dict:

    # Winter = W, Spring = P, SUmer = S, Fall = F
    year = {  "WJanuary": [[] for _ in range(31)], 
                "WFebruary": [[] for _ in range(28)],
                "PMarch": [[] for _ in range(31)],
                "PApril": [[] for _ in range(30)], 
                "PMay": [[] for _ in range(31)],
                "SJune": [[] for _ in range(30)],
                "SJuly": [[] for _ in range(31)],
                "SAugust": [[] for _ in range(31)],
                "FSeptember": [[] for _ in range(30)],
                "FOctober": [[] for _ in range(31)],
                "FNovember": [[] for _ in range(30)],
                "WDecember": [[] for _ in range(31)]
            }
    
    temp_change = args["temp_change"]
    winter_temp = args["winter"]
    spring_temp = args["spring"]
    summer_temp = args["summer"]
    fall_temp = args["fall"]
    
    fc_change = args["fc_change"]
    winter_fc = args["winterfc"]
    spring_fc = args["springfc"]
    summer_fc = args["summerfc"]
    fall_fc = args["fallfc"]

    verical_angle = 45
    horizontal_angle = 0

    for k, v in year.items():
        if (k[0] == "W"):
            for i in range(len(v)):
                # [verical_angle, horizontal_angle, fc,  temp]
                v[i].append(verical_angle)
                v[i].append(horizontal_angle)
                v[i].append(random.randint(winter_fc - fc_change, winter_fc + fc_change))
                v[i].append(random.randint(winter_temp - temp_change, winter_temp + temp_change))
        elif (k[0] == "P"):
            for i in range(len(v)):
                v[i].append(verical_angle)
                v[i].append(horizontal_angle)
                v[i].append(random.randint(spring_fc - fc_change, spring_fc + fc_change))
                v[i].append(random.randint(spring_temp - temp_change, spring_temp + temp_change))
        elif (k[0] == "S"):
            for i in range(len(v)):
                v[i].append(verical_angle)
                v[i].append(horizontal_angle)
                v[i].append(random.randint(summer_fc - fc_change, summer_fc + fc_change))
                v[i].append(random.randint(summer_temp - temp_change, summer_temp + temp_change))
        elif (k[0] == "F"):
            for i in range(len(v)):
                v[i].append(verical_angle)
                v[i].append(horizontal_angle)
                v[i].append(random.randint(fall_fc - fc_change, fall_fc + fc_change))
                v[i].append(random.randint(fall_temp - temp_change, fall_temp + temp_change))

    return year


# weather = generate_weather_data(  temp_change = 13,
#                         winter = 30,
#                         spring = 68,
#                         summer = 88,
#                         fall = 60,
#                         fc_change = 800,
#                         winterfc = 900,
#                         springfc = 1500,
#                         summerfc = 2000,
#                         fallfc = 1300
#                              )

# pr(weather)

if (__name__ == "__main__1"):
    weather = generate_weather_data(  temp_change = 13,
                        winter = 30,
                        spring = 68,
                        summer = 88,
                        fall = 60,
                        fc_change = 800,
                        winterfc = 900,
                        springfc = 1500,
                        summerfc = 2000,
                        fallfc = 1300
                             )
    
    columns = ["Vertical Angle", "Horizontal Angle", "Fc", "TempF", "DCV Output"]
    raw_dataset = pd.read_csv("Solar Panel Data - Sheet1.csv", names=columns)

    # sorted_dataset = raw_dataset.sort_values(by=[columns[4]], ascending=True)
    # pr(dataset)
    data_table = convertDataFrame(raw_dataset)
    df_raw = pd.DataFrame.from_dict(data_table)
    df = df_raw.copy()
    #print(df)

    model = Polynomial_Prediction_Model(df, columns[4], 4)
    model.compile()
    y_prediction = model.predict(model.input_features)
    # print(model.input_features.shape)
    # model.plot(x = model.df["Fc"], y = y_prediction)

    loss, avg_loss = model.calc_loss(y_prediction)
    print(loss)
    loss.append(avg_loss)
    print(f"Avg loss: {avg_loss}")
    #model.plot_loss(loss)
    # print()
    print(model.predict(np.array([[45, 0, 2000, 30]])))
    # print(model.model.coef_)
    # print(model.model.intercept_)


    plt.show()
