# Science Fair Solar Panel Research Project

## Research Question
How accurately can a predictive model estimate the energy output of a solar panel based on the angle of incoming sunlight, solar intensity, temperature, and panel position?

Polynomial regression is a supervised machine learning method that is capable of modeling a non-linear relationship between a set of inputs and outputs.  

## BackGround Research
Predicting a solar panel's energy output also known as solar power forecasting is important in maintaining a stable energy grid. Solar predicting is one of the most important parts of advanced energy management systems (EMS) because it can prevent energy shortages or even blackouts (Gridx, 2024).  There are various types of solar forecasting methods such as physical, statistical, Holistic and integrated methods (Haoyin Ye, 2022). The  method I’m using for this project will be a polynomial regression predictive model. Polynomial regression is similar to linear regression but it can model non-linear relationships using exponents. The main goal is to model the expected dependent variable (y) and it’s one of the most basic types of machine learning (geeksforgeeks, 2024). My polynomial regression model will model as a 4th degree polynomial, this means the largest exponent is a 4. For this project I am using sk learn which is a free and open source machine learning library in Python (scikit-learn, 2024). Some other useful sources/documentations were on statology.org (Zach Bobbitt, 2020). 


## Hypothesis 
A polynomial regression model can be used to accurately predict the energy output of a solar panel under various conditions. 


## Procedure and Materials: 
Gather data on the solar panel under different conditions.
Create a polynomial regression model in python.
Train model on collected data.
Get historical weather data on different regions of the United States 
Use the model to determine which region will have the best average solar panel efficiency throughout the whole year.

Solar panel
Photometer 
Voltage meter 
Thermometer
Wires

Results: loss  value graph and data graph


## Analysis 1
The bar graph shows the average loss value sorted by light intensity. The loss value is calculated by taking the difference of the predicted value and actual value (predicted  - actual). The loss value is useful for determining the accuracy of the prediction model. On the graph, the y axis shows loss/expected error and the x axis shows the light intensity categories ranging from low, medium and high. The last bar on the graph labeled “Total Avg Loss” doesn’t fall under any light intensity category and is the sum of the average loss values across all light intensities.

 ## Analysis 2
The scatter plot shows the relationship between light intensity (Fc) on the x-axis and energy output (DCV) on the y-axis. Each point on the graph is color-coded according to the color bar displayed on the right. Blue represents cold temperatures below 55°F, green corresponds to mid-range temperatures (around room temperature), and red indicates high temperatures.
Additionally, the three lines on the graph are also color-coded using the same color scheme. The blue line represents the line of best fit for data points below room temperature, the green line corresponds to the line of best fit at room temperature, and the red line represents the line of best fit for points above room temperature.
Comparing the lines, the blue line demonstrates the highest energy output, followed by the green line, which exhibits a higher energy output than the red line.


## Conclusion/Discussion
Based on the bar graph, the total average loss value is 0.13. This means that when data is input and a prediction is made, the expected error is approximately ±0.13. For example, if the predicted output is 5 V, the actual output can be expected to fall within the range of 4.87 V to 5.13 V.  





The scatter plot helps visualize the relationship between light intensity, temperature and energy output. The three lines that are categorized based on temperature help illustrate the effects temperature can have on the energy output  of a solar panel. The blue line which is higher than the other two has the optimal energy output. In fact the highest recorded output is at 31 degrees (F) Which means . The red line shows that hotter temperatures negatively affect the solar panels output. 

## Further Research




https://www.gridx.ai/knowledge/what-is-solar-power-forecasting#solar-forecasting-methodsnbsp

https://www.statology.org/polynomial-regression-python/

https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.875790/full

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
