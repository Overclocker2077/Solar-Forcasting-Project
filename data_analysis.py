import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print as pr

np.set_printoptions(precision=3, suppress=True)
sns.set_theme()


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
def split_filter(dict, target_index, key_index, split_func = lambda x : x > 65) -> tuple[list[float], list[float]]:
    array1 : list[float] = []
    array2 : list[float] = []
    for i in range(len(dict[target_index])):
        if (split_func(dict[key_index][i])):
            array1.append(dict[target_index][i])
        else:
            array2.append(dict[target_index][i])
    return array1, array2

columns = ["Vertical Angle", "Horizontal Angle", "Fc", "TempF", "DCV Output"]
raw_dataset = pd.read_csv("Solar Panel Data - Sheet1.csv", names=columns)
# sorted_dataset = raw_dataset.sort_values(by=[columns[4]], ascending=True)
# pr(dataset)
data_table = convertDataFrame(raw_dataset)

"""# pr(columns)
# pr(data_table)

# 3 dimensional plot     temp, fc, dcv
# sns.stripplot(x=data_table[columns[2]], y = data_table[columns[3]], hue = data_table[columns[4]])

# 2 dimensional plot fc, dcv
# sns.stripplot(x=data_table[columns[2]], y = data_table[columns[4]], hue=data_table[columns[3]])
# # raw_dataset.plot.bar(x=columns[0], y = columns[4])"""

def Polynomial_regression(degree: int = 1) -> tuple[np.poly1d, np.poly1d, np.ndarray, np.ndarray]:
    fc_above, fc_below = split_filter(dict = data_table, target_index=columns[2], key_index=columns[3])
    dcv_above, dcv_below = split_filter(dict = data_table, target_index=columns[4], key_index=columns[3])
    
    model_above_temp = np.poly1d(np.polyfit(x = fc_above,  y = dcv_above, deg = degree))
    polyline_above_temp = np.linspace(1, 4000, 50)

    model_below_temp = np.poly1d(np.polyfit(x = fc_below,  y = dcv_below, deg = degree))
    polyline_below_temp = np.linspace(1, 4000, 50)

    return model_below_temp, model_above_temp, polyline_below_temp, polyline_above_temp

def Logarithmic_regression(degree: int = 1) -> tuple[np.poly1d, np.poly1d, np.ndarray, np.ndarray]:
    fc_above, fc_below = split_filter(dict = data_table, target_index=columns[2], key_index=columns[3])
    dcv_above, dcv_below = split_filter(dict = data_table, target_index=columns[4], key_index=columns[3])
    
    model_above_temp = np.poly1d(np.polyfit(x = np.log(fc_above),  y = dcv_above, deg = degree))
    polyline_above_temp = np.linspace(1, 4000, 50)

    model_below_temp = np.poly1d(np.polyfit(x = np.log(fc_below),  y = dcv_below, deg = degree))
    polyline_below_temp = np.linspace(1, 4000, 50)

    return model_below_temp, model_above_temp, polyline_below_temp, polyline_above_temp


def plot_data():
    model_below_temp, model_above_temp, polyline_below_temp, polyline_above_temp = Polynomial_regression(4)
    
    #model_below_temp, model_above_temp, polyline_below_temp, polyline_above_temp = Logarithmic_4th_degree()

    plt.scatter(x = data_table[columns[2]], y = data_table[columns[4]], label='Data', c = data_table[columns[3]])
    plt.plot(polyline_above_temp, model_above_temp(polyline_above_temp), color = "Orange")
    plt.plot(polyline_below_temp, model_below_temp(polyline_below_temp), color = "Blue")
    print(model_below_temp)
    print(model_above_temp)
    plt.colorbar()
    plt.xlabel('Fc')
    plt.ylabel('DCV Output')
    plt.legend()
    plt.grid(True)

plot_data()

plt.title('Solar Panel Data') 
plt.show()

