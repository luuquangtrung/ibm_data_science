# COURSE 6: DATA ANALYSIS WITH PYTHON


Libraries used for this course: `matplotlib`, `pandas`, and `scikit-learn`


# WEEK 1: Importing Datasets

## 1.1. The Problem

Why Data Analysis?
* Data is everywhere
* DA/DS helps us answer questions from data
* DA plays an important role in:
	* Discovering useful info
	* Answering questions
	* Predicting future or the unknown

Dataset: `csv` file (comma-separated values)
* Each column is a feature or target

**Example:** Estimating used car prices:
* Is there data on the prices of other cars and their characteristics? 
* What features of cars affect their prices? Color? Brand? Housepower? Something else?
* Does horsepower also effect the selling price, or perhaps something else? 
* Asking the right questions in terms of data


## 1.2. Understanding the Data

**Example:** Estimating used car prices (cont.)
Attributes:
* Symboling: corresponds to the insurance risk level of a car
* Normalized losses
* Make, i.e., the manufacturer (car brand, e.g., audi, bmw, etc.)
* Prices: `target value` or `label`


## 1.3. Python Packages for Data Science

Three groups of Python data analysis libraries:
1. Scientific computing libraries: 
	* `pandas`: data structures and tools. Primary instrument is dataframe
	* `numpy`: array ans matrices
	* `scipy`: integrals, solving differential equations, optimization

2. Visualization libraries: 
	* `matplotlib`: plot & graph (most popular)
	* `seaborn`: heat maps, time series, violin plots

3. Algorithmic libraries:
	* `sciki-learn`: ML: regression, classification
	* `statmodels`: explore data, estimate stat. models, perform stat. tests


## 1.4. Importing and Exporting Data in Python

Importing data - 2 important properties:
* Format: `csv`, `json`, `xlsx`, `hdf`, etc.
* File path: local or online

Using `pandas`:

```python
import pandas as pd
url = "https://..../import-85.data"
df = pd.read_csv(url)	# including header
df = pd.read_csv(url, header = None) # no header

# Printing the data frame
df.head(n)	# show the first n rows
df.tail(n)	# show last n rows

# Replace default header by the list
headers = ["symboling", "normalized-losses"]
df.columns = headers

# Export a pandas dataframe to CSV:
path = "C:\automobile.csv"
df.to_csv(path)

```

**Summary**

data_format	| read 				| save
------------|-------------------|-----------------
`csv`		| `pd.read_csv()`	| `df.to_csv()`
`json` 		| `pd.read_json()`	| `df.to_json()`
`excel` 	| `pd.read_excel()`	| `df.to_excel()`
`sql` 		| `pd.read_sql()`	| `df.to_sql()`


## 1.5. Getting Started Analyzing Data in Python

Basic insights from the data:
* Understand your data before you begin any analysis
* Should check: 
	* Data types
	* Data distribution
* Locate potential issues with the data, e.g., wrong types
* Why check data types?
	* Potential info and type mismatch
	* Compability with python methods

Data Types:

panda types 				| native python type 		| description
----------------------------|---------------------------|------------------------
object 						| string 					| numbers and strings
int64 						| int 						| numeric characters
float64 					| float 					| numberic with decimals
datetime64, timedelta[ns] 	| N/A (1)					| time data

(1) but see `datetime` module of Python std lib


```python
df.dtypes 		# return the list of data types in df

# Show statistical summary of df (count, mean, std, min, 25-50-75%, max)
df.describe()	

'''
By default, the dataframe.describe functions skips rows and columns that do not contain numbers. It is possible to make the describe method worked for object type columns as well
'''

# Show full statistical summary: = decribe() + unique, top, freq
df.describe(include = "all") 

'''
Some values may appear as NaN which stands for not a number. This is because that particular statistical metric cannot be calculated for that specific column data type
'''

# Show concise (short) summary
df.info() # show top 30 rows and bottom 30 rows of the df

```




## 1.6. Copy of Getting Started Analyzing Data in Python


## LAB 1: Importing Datasets

## QUIZ: Importing Datasets




# WEEK 2: Data Wrangling

## 2.1. Data Pre-processing

**Data Pre-processing:**
* Process of converting or mapping data from one raw form into another format to make it ready for further analysis.
* Also known as: `Data Cleaning` or `Data Wrangling`

**Outline:**
1. Identify and handle missing values
2. Data formatting: Use some methods in Python Pandas that can standardize the values into the same format, or unit, or convention
3. Data normalization (centering/scaling): bring all data into a similar range for more useful comparison. Specifically, we'll focus on the techniques of centering and scaling
4. Data binning: creates bigger categories from a set of numerical values
5. Turning categorical values to numeric variables: to make statistical modeling easier

Example: Simple dataframe operations:
```python
df["symboling"] = df["symboling"] + 1 	# adding 1 to the current values

```

## 2.2. Dealing with Missing Values in Python

Missing values could be represented as: `"?"`, `"N/A"`, `0`, or just a blank cell

How to deal with missing data?

1. Check with the data collection source
2. Drop the missing values
	* Drop the variable
	* Drop the data entry
3. Replace the missing values
	* Replace it with an average (of similar datapoints)
	* Replace it by frequency, e.g., in case the variable is not numeric but categorical
		* Example: For a variable like `"fuel"` type, there isn't an average fuel type since the variable values are not numbers. In this case, one possibility is to try using the **mode**, the most common like gasoline `"gasoline"`
	* Replace it based on other functions
		* For example, he may know that the missing values tend to be old cars and the normalized losses of old cars are significantly higher than the average vehicle.
4. Simply leave it as missing data

**Summary:** To deal with missing values for categorical data
* replace the missing value with the mode of the particular column
* replace the missing value with the value that appears most often of the particular column

```python
'''
Drop NA values in dataframe
axis = 0: drop the entire row
axis = 1: drop the entire col
inline = True:  just writes the result back into the data frame
'''

df.dropna(axis = 0)	# drop all rows containing NaN 

# Adding condition
df.dropna(subset=["price"], axis=0, inplace=True)

# The above is equivalent to this below code:
df = df.dropna(subset=["price"], axis=0)


'''
Replace mising values:
df.replace(missing_value, new_value)
'''

# Example: Replace NA by mean
mean = df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean, inplace=True)

``` 


## 2.3. Data Formatting in Python


```python

df["city-mpg"] = 235/df[
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True) # rename df column names

# Convert object to integer:
df.dtypes() 		# to check the type of df
df["price"] = df["price"].astypes("int")	# to convert df to integer

```


## 2.4. Data Normalization in Python (centering/scaling)

Data normalization: Uniform the features value with different range


Not-normalized					| Normalized
--------------------------------|------------------------------
Different range 				| Similar value range
Hard to compare 				| Similar intrinsic influence 
Variable with higher values  	| on analytical model	
will influence the result more 	|


**Example:**

<table>
	<tr>
		<td colspan="2">Not-normalize</td>
		<td colspan="2">Normalize</td>
	</tr>
	<tr>
		<td>age</td>
		<td>income</td>
		<td>age</td>
		<td>income</td>
	</tr>
	<tr>
		<td>30</td>
		<td>10,000</td>
		<td>0.2</td>
		<td>0.2</td>
	</tr>
	<tr>
		<td>20</td>
		<td>20,000</td>
		<td>0.3</td>
		<td>0.04</td>
	</tr>
	<tr>
		<td>40</td>
		<td>50,000</td>
		<td>0.4</td>
		<td>1</td>
	</tr>
</table>

**Normalization methods:**
1. *Simple Feature Scaling*: `x_new = x_old/x_max`, results range in `[0, 1]`
2. *Min-max*: `x_new = (x_old - x_min)/(x_max - x_min)`, results range in `[0, 1]`
3. *Z-score (standard score)*: `x_new = (x_old - mu)/sigma`, where (mu, sigma): average and standard deviation of the feature. Results hover around `0`, and typically range between `-3` and `+3` but can be higher or lower


```python
# Simple feature scaling
df["age"] = df["age"]/df["age"].max()	

# Min-max
df["age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
			
# Z-score
df["age"] = (df["age"] - df["age"].mean()) / df["age"].std() 

```


## 2.5. Binning in Python

Binning:
* Grouping of values into `bins`
* Converts numeric into categorical variables
* Group a set of numeric variables into a set of `bins`

Examples: 
* Bin age into `0-5`, `6-10`, `11-15`, and so on
* Bin prices (e.g., 5000; 10,000;... 39,000; 44,500, etc.) into categories: `low`, `mid`, `high`


**NOTE:** 
* Sometimes binning can improve accuracy of the predictive models
* In addition, sometimes we use data binning to group a set of numerical values into a smaller number of bins to have a better understanding of the data distribution


```python
binwidth = int((max(df["price"]) - min(df["price"]))/4)
bins = range(min(df["price"]), max(df["price"]), binwidth)

group_names = ['low', 'medium', 'high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)

# And then we can visualize binned data by histograms

```

## 2.6. Turning categorical variables into quantitative variables in Python

Problem:
* Most statistical models cannot take in objects/strings as input 

Categorical to numeric solution:
* Add dummy variables for each unique category
* Assign `0` or `1` in each category (called **One-hot Encoding**)

Example:

Car | Fuel 		| gas 	| diesel
----|-----------|-------|--------
A 	| gas 		| 1 	| 0
B 	| diesel 	| 0 	| 1



```python
'''
Use pandas.get_dummies() method
Convert categorical variables to dummy variables (0 or 1)
'''
pd.get_dummies(df["fuel"])

df['length'] = df['length']/df['length'].max()
df['length'] = df['length']/df['length'].max()
```




# WEEK 3: Exploratory Data Analysis (EDA)

* Preliminary step in DA to:
	* Summarize main characteristics of the data
	* Gain better understanding of the data set
	* Uncover relationships between variables
	* Extract important variables
* Question: "What're the characteristics that have the most impact on the car price?"
* Learning objectives:
	* Descriptive statistics: Describe basic features of a data set, and obtain a short summary about the sample and measures of the data
	* GroupBy: Grouping data -> transform the dataset
	* ANOVA: Analysis of variance, a statistical method in which the variation in a set of observations is divided into distinct components
	* Correlation between different variables
	* Advance correlation: various correlations statistical methods namely, Pearson correlation, and correlation heatmaps

## 3.1. Descriptive Statistics
* Describe basic features of data
* Giving short summaries about the sample and measures of the data

### Method: `df.describe()`
* Summarize statistics using `pandas` `describe()` method

```python
df.describe()

'''
- Show stats of dataframe: count, mean, std, min, 25%, 75%, max
- NaN is automatically skipped 
'''
```

### Method: `df.value_count()`
* Summarize the categorical data is
* Return a Series containing counts of unique values.
* The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.

```python
drive_wheels_counts = df["drive-wheels"].value_count()

# Change column name
drive_wheels_counts.rename(columns = {'drive-wheels': 'value_count' inplace=True})
drive_wheels_counts.index.name = 'drive-wheels'
```

### Visualize using Box Plot
* Great way to visualize numeric data, since you can visualize the various distributions of the data
* The main features of the box plot shows are:
	* **Median** of the data which represents where the middle data point is
	* **Upper quartile** shows where the 75th percentile is
	* **Lower quartile** shows where the 25th percentile is
	* The data between the upper and lower quartile represents the **interquartile range**
	* **Lower extreme** and **upper extreme**: These are calculated as 1.5 times the interquartilre range above the 75th percentile and as 1.5 times the IQR below the 25th percentile
	* Box plots also display outliers as individual dots that occur outside the upper and lower extremes
* With box plots, you can easily spot outliers and also see the *distribution* and *skewness* of the data. Box plots make it easy to compare between groups.

```python
sns.boxplot(x='drive-wheels', y='price', data=df)
```

### Visualize using Scatter Plot
* Show the relationship between two variables
* Each observation represented as a point
* Predictor/independent variables on x-axis (*horizontal*)
* Target/dependent variables on y-axis (*vertical*)

```python
x = df["engine-size"]
y = df["price"]
plt.scatter(x,y)
```


## 3.2. `GroupBy`to group data in `pandas`
* Can be applied on categorical variables
* Group data into categories
* Single or multiple variables

Example:
```python
# First pick out the three data columns we are interested in
df_test = df[['drive-wheels', 'body-style', 'price']]

# Group the reduced data according to 'drive-wheels' and 'body-style' 
# Since we are interested in knowing how the average price differs 
# across the board, we can take the mean of each group and append 
# it this bit at the very end of the line too
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

# Use the groupby function to find the average "price" of each car based on "body-style"
df[['price','body-style']].groupby(['body-style'],as_index= False).mean()
```

Result:
no.	| drive-wheels 	| body-style 	| price
----|---------------|---------------|------------
0	| 4wd			| hatchback		| **7603.00**
1	| 4wd			| sedan			| 12647
2	| 4wd			| wagon			| 9095
3	| fwd			| convertible	| 11595
4	| fwd			| hardtop		| 8249
5	| fwd			| hatchback		| 8396
6	| fwd			| sedan			| 9811
7	| fwd			| wagon			| 9997
8	| rwd			| convertible	| 23949
9	| rwd			| hardtop		| 24202
10	| rwd			| hatchback		| 14337
11	| rwd			| sedan			| 21711
12	| rwd			| wagon			| 16994

* According to our data, rear wheel drive (rwd) convertibles and rear wheel drive hard hardtops have the highest value while four wheel drive (4wd) hatchbacks have the lowest value
* A table of this form isn't the easiest to read and also not very easy to visualize
* To make it easier to understand, we can transform this table to a pivot table by using the **pivot** method


## Pandas method: `pivot()`
* Typical `pivot` table has:
	* One variable displayed along the columns 
	* One other variable displayed along the rows

```python
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
# body-style: along the columns, drive-wheels: along the rows
```
Result:

				| price 		| 		    |		    | 		|
----------------|---------------|-----------|-----------|-------|--------
body-style		| convertible 	| hardtop 	| hatchback | sedan	| wagon
drive-wheels 	|  				|  		  	| 		  	| 		| 
4wd				| 20239 		| 20239 	| 7603 		| 12647	| 9095
fwd				| 11595 		| 8249	  	| 8396 		| 9811	| 9997
rwd				| 23949 		| 24202 	| 14337 	| 21711	| 16994

## Heatmap
* Plot target variable over multiple variables
```python
plt.pcolor(df_pivot, cmap='RdBu') # Red-Bkye color scheme
plt.colorbar()
plt.show()
```


## 3.3. Correlation

### Definition:
* Correlation is a  statistical metric for measuring to what extent different variables are interdependent
* In other words, when we look at two variables over time, if one variable changes how does this affect change in the other variable?

Examples:
* Lung cancer <-> Smoking
* Rain <-> Umbrella

**Attention:**
* Correlation does not imply causation, i.e., we can say that umbrella and rain are correlated but we would not have enough information to say whether the umbrella caused the rain or the rain caused the umbrella

### Correlation - Positive linear relationship
```python
# Correlation between two features ('engine-size' and 'price')
sns.regplot(x='engine-size', y='prices', data=df)
plt.ylim(0,)
```

![c6_w3_correlation_positive](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_correlation_positive.png)


### Correlation - Negative linear relationship
```python
# Correlation between two features ('highway-mpg' and 'price')
sns.regplot(x='highway-mpg', y='prices', data=df)
plt.ylim(0,)
```

![c6_w3_correlation_negative](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_correlation_negative.png)


### Weak Correlation
```python
# Correlation between two features ('peak-rmp' and 'price')
sns.regplot(x='peak-rmp', y='prices', data=df)
plt.ylim(0,)
```

![c6_w3_correlation_weak](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_correlation_weak.png)


## 3.3. Pearson Correlation
* Measure the strength of the correlation between 2 features:
	1. Pearson correlation coefficient
		* Close to `+1`: large positive relationship
		* Close to `-1`: Large negative relationship
		* Close to `0`: No relationship
	2. `P-value`
		* P-value < 0.001: *Strong* certainty in the result
		* P-value < 0.05: *Moderate* certainty in the result
		* P-value < 0.1: *Weak* certainty in the result
		* P-value > 0.001: *No* certainty in the result
* Strong correlation:
	* Correlation coefficient close to `1` or `-1`
	* P-value < 0.001


![c6_w3_correlation_pearson](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_correlation_pearson.png)


Calculate the Pearson correlation between `'horsepower'` and `'price'`:
```python
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
# pearson_coef = 0.81: close to 1 -> strong positive correlation
# p_value = 9.35e-48 << 0.001 indicates: stong certainty in the result
```

Taking all variables into account, we can now create a heatmap that indicates the correlation between each of the variables with one another


![c6_w3_correlation_pearson_heatmap](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_correlation_pearson_heatmap.png)

* The color scheme indicates the Pearson Correlation coefficient, indicating the strength of the correlation between two variables
* We can see a diagonal line with a dark red color indicating that all the values on this diagonal are highly correlated
* This makes sense because when you look closer, the values on the diagonal are the correlation of all variables with themselves, which will be always one
* This correlation heatmap gives us a good overview of how the different variables are related to one another, and most importantly, how these variables are related to price


## 3.4. ANOVA: Analysis of Variance
* Statisticcal comparison of groups
* Example: average price of different vehicle makes. The question we may ask is how different categories the make feature as a categorical variable has impact on the price?

![c6_w3_anova](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_anova.png)

* Why perform ANOVA?
	* Finding correlation between different groups of a categorical variable
* What we obtain from ANOVA?
	* `F-test` score: variation between sample group means divided by variation within sample group
	* Small/large F implies poor/strong correlation between variable categories and target variable

![c6_w3_f_test_small](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_f_test_small.png)

![c6_w3_f_test_large](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w3_f_test_large.png)


Example: ANOVA between `honda` and `subaru`

```python
df_anova = df[["make", "price"]]
grouped_anova = df_anova.groupby(["make"])
anova_result = stats.f_oneway(grouped_anova.get_group("honda")["price"], \
	grouped_anova.get_group("subaru")["price"])

'''
ANOVA results:
F = 0.19744031275
p = F_onewayResult(statistic=0.1974403127, p_value=0.660947824)
The prices between Hondas and Subarus are not significantly different, 
as the F-test score is less than one, and p-value is larger than 0.05
'''
```

# WEEK 4: Model Development

## 4.1. Model Development

* A model can be thought of as a mathematical equation used to predict a value given one or more other values
* Relatinng one or more independent variables to dependendt variables
* Usually the more relevant data you have the more accurate your model is

Example:

![c6_w4_model_development](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_model_development.png)

## 4.2. Linear Regression and Multiple Linear Regression

* Linear Regression: 1 independent variable ($x_1$) -> LR -> Prediction
* Multiple Linear Regression: multi independent variables ($x_1, ..., x_d$) -> MLR -> Prediction

### Simple Linear Regression: $$y = b_0 + b_1*x$$

* `x`: predictor (independent) variable
* `y`: target (dependent) variable
* `b_0`: intercept
* `b_1`: slope

```python
# Import linear_model from scikit-learn
from sklearn.linear_model import LinearRegression

# Create a Linear Regression object using the constructor
lm = LinearRegression()

# Define the predictor variable and target variable
X = df[['highway-mpg']]
Y = df['price']

# Then use lm.fit(X,Y) to fit the model, i.e., find the parameters b0 and b1
lm.fit(X,Y)

# Then obtain a prediction
Yhat = lm.predict(X) # Yhat is an array having same number of samples as input X 

# Get b0
lm.intercept_

# Get b1
lm.coef_

# Relationship between Price and Highway MPG is given by
Price = lm.intercept_ - lm.coef_*highway-mpg
```

### Multiple Linear Regression (MLR): $$\hat(Y) = b_0 + b_1*x_1 + b_2*x_2 + ...$$
MLR is used to explain relationship between 
* 1 continuous target (`Y`) variable
* 2 or more predictor (`X`) variables 
* `b_0`: intercept
* `b_i`: coefficient or parameter of `x_i` (`i = 1, 2, 3,...`)

```python
# Extract 4 predictor variables and store them in the variable Z  
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Train the model as before
lm.fit(Z, df['price'])

# Obtain prediction
Yhat = lm.predict(X)

# MLR - Estimated Linear Model
lm.intercept_ 	# b0
lm.coef_ 		# coefficients array: array([b1, b2, b3, b4])
```

## 4.3. Model Evaluation using Visualization

### Regression plot: 
* Gives us a good estimate of 
	* Relationship between 2 variables
	* Strength of correlation
	* Direction of the relationship (positive or negative)
* Shows us a combination of 
	* Scatter plot: where each point represent a different `y`
	* The fitted linear regression line `Yhat`

Example:

```python
import seaborn as sns

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
``` 

![c6_w4_regresison_plot_1](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_regresison_plot_1.png)

### Residual plot:
* Represents the error between the actual value
* Examining the predicted value and actual value we see a difference. We obtain that value by subtracting the predicted value, and the actual target value. We then plot that value on the vertical axis with the dependent variable as the horizontal axis

Look at the **spread of the residual**:

Example 1: 
![c6_w4_residual_plot_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_residual_plot_2.png)
* Randomly spread out around 

Example 2:
![c6_w4_residual_plot_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_residual_plot_3.png)
* Not randomly spread out around the x-axis
* Non-linear model may be more appropriate

Example 3:
![c6_w4_residual_plot_4](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_residual_plot_4.png)
* Not randomly spread out around the x-axis
* Variance appears to change with x-axis

### Create Residual Plot with Seaborn:

```python
import seaborn as sns
sns.residplot(df['highway-mpg'], df['price'])
``` 


### Distribution plot 
* Counts the predicted value versus the actual value
* These plots are extremely useful for visualizing models with more than one independent variable or feature. 

![c6_w4_distribution_plot_1](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_distribution_plot_1.png)

Compare the distribution plots:
* The fitted values that result from the model
* The actual values

![c6_w4_distribution_plot_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_distribution_plot_2.png)


```python
import seaborn as sns

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Value", ax=ax1)
``` 


## 4.4. Polynomial Regression and Pipelines

### Polynomial regression:
* Is a special case of the general linear regression
* This method is beneficial for describing curvilinear relationships
* Curvilinear relationship: It's what you get by squaring or setting higher-order terms of the predictor variables in the model transforming the data (The model can be quadratic which means that the predictor variable in the model is squared)

### Polynomial order:
* Quadratic - 2nd order: `Yhat = b_0 + b_1*x_1 + b_2*(x_1)^2`
* Cubic - 3rd order: `Yhat = b_0 + b_1*x_1 + b_2*(x_1)^2 + b_3*(x_1)^3`
* Higher order: idem.

### Polynomial Regression with numpy:

```python
# Calculate polynomial of 3rd order
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

# Print out the model
print(p)	# -1.557(x1)^3 + 204.8(x1)^2+ 8965x_1 + 1.37e5
``` 

Polynomial regression with more than 1 dimension:
* Form: `Yhat = b0 + b1*X1 + b2*X2 + b3*X1*X2 + b4(X1)^2 + b5(X2)^2 + ...`
* We can not only use np.polyfit, we must do the "pre-processing" using `scikit-learn`

```python
# Preprocessing library in scikit-learn
from sklearn.preprocessing import PolynomialFeatures

# Create a polynomial feature object. 
# The constructor takes the degree of the polynomial as a parameter
pr = PolynomialFeatures(degrees=2, include_bias=False)

# Transform the features into a polynomial feature with the fit_transform method
x_poly = pr.fit_transform(x['horsepower', 'curb-weight'])

``` 

### Pre-processing
Normalize each feature simultaneously:

```python
from sklearn.preprocessing import StandardScaler

SCALE = StandardScaler()
SCALE.fit(x_data['horsepower', 'highway-mpg'])
x_scale = SCALE.transform(x_data[['horsepower', 'highway-mpg']])
``` 

### Pipelines
* There are more normalization methods available in the preprocessing library as well as other transformations
* For instance, we can simplify our code by using a pipeline library:
	* There are many steps to getting a prediction. For example: (1) Normalization -> (2) Polynomial Transform -> (3) Linear Regression
	* We simplify the process using a pipeline
	* Pipelines sequentially perform a series of transformations (1 and 2). The last step (3) carries out a prediction

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

'''
create a list of tuples
* First element: name of the estimator model: 'scale', 'polynomial', 'model'
* Second element: contains model constructor: StandardScaler(), etc.
'''
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degrees=2)), ... ('model', LinearRegression())]

# Input the list in the pipeline constructor
pipe = Pipeline(Input) # A Pipeline object

# Now we can train the pipeline by applying the train method to the pipeline 
# object. We can also produce a prediction as well. The method normalizes 
# the data, performs a polynomial transform, then outputs of prediction.

pipe.train(X['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'], y)
yhat = Pipe.predict(X['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'])

# X -> Normalization -> Polynomial Transform -> Linear Regression -> yhat
``` 

## 4.5. Measures for In-Sample Evaluation

* A way to numerically determine how good the model fits on dataset
* 2 important measures to determine the fit of a model:
	* Mean Square Error (MSE)
	* R-squared (R^2)

### Mean Square Error

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['price'], Y_predict_simple_fit)
```

![c6_w4_mse](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_mse.png)

### R-Squared (`R^2`)

* `R^2` = Coefficient of Determination
* Is a measure to determine how close the data is to the fitted regression line
* `R^2`: The percentage of variation of the target variable (`Y`) that is explained by the linear model
* Think about it as comparing a regression model to a simple model, i.e., the mean of the data points
* R^2 = 1 - (MSE of the regression line)/(MSE of the average of the data [or ybar])
* For the most part, `R^2` takes values in `[0,1]`

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(df['price'], Y_predict_simple_fit)
```
Example: `ybar = 6`
![c6_w4_rsquared](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_rsquared.png)

Example:
![c6_w4_rsquared_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w4_rsquared_2.png)

* Blue line represents the regression line
* Blue squares represent the MSE of the regression line
* Red line represents the average value of the data points
* Red squares represent the MSE of the red line
* Comment: 
	* The area of the blue squares is much smaller than the area of the red squares
	* In this case, because the line is a good fit, the `MSE` is small, therefore the numerator is small. The `MSE` of the line is relatively large, as such the numerator is large. 
	* A small number divided by a larger number is an even smaller number. Taken to an extreme this value tends to `0`. If we Plug in this value from the previous slide for `R^2`, we get a value near one, this means the line is a good fit for the data

```python
X = df[['highway-mpg']]
Y = df['price']

lm.fit(X, Y) 

# Calculate R^2
lm.score(X, Y) # 0.4965...

```

Comment: If `R^2` value is negative, it can be due to overfitting

## 4.6. Prediction and Decision Making

Determine if our model is correct? 
* Make sure your model results (predicted values) make sense
* Use visualization: regression plot, residual plot, distribution plot (good for multiple linear regression), etc.
* Use numerical measures for evaluation: `MSE`, `R^2`. Acceptable value for R-squared depends on what field you're studying. Some authors suggest a value should be equal to or greater than 0.10 (Falk and Miller, 1992)
* Comparing between different models
	* For instance: Compare MLR and SLR:
	1. Is a lower MSE always implying a better fit?
		* Not necessarily
	2. MSE for an MLR model will be smaller than the MSE for an SLR model, since the errors of the data will decrease when more variables are included in the model
	3. Polynomial regression will also have a smaller MSE than regular regression
	4. A similar inverse relationship holds for R-squared


# WEEK 5: Model Evaluation


## 5.1. Model Evaluation and Refinement
* In-sample evaluation: tells us *how well our model will fit the data used to train it*
* Problem: It does not tell us *how well the trained model can be used to predict new data*
* Solution: Split our data up, use the in-sample data or training data to train the model. 
	* The rest of the data, called Test Data, is used as out-of-sample data
	* This data is then used to approximate how the model performs in the real world
	* Separating data into training and testing sets is an important part of model evaluation
	* We use the test data to get an idea how our model will perform in the real world. When we split a dataset, usually the larger portion of data is used for training and a smaller part is used for testing
	* For example, we can use 70% of the data for training, and 30% for testing. 
	* **Training set** (in-sample data): To build a model and discover predictive relationships
	* **Testing set** (out-of-sample data): To evaluate model performance
	* When we have completed testing our model, we should use all the data to train the model.
* In practical, use the `scikit-learn`'s function `train_test_split()` 

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

'''
x_data: features or independent variables
y_data: dataset target: df['price']
x_train, y_train: parts of available data as training set
test_size: percentage of data for training (30% here)
random_state: nb of generators used for random sampling
'''
```

### Generalization Performance
* Generalization error is measure of how well our data at predicting previously unseen data
* The error we obtain using our testing data is an approximation of this error
* Problem:
	* If we experiment again training the model with a different combination of samples, we also get a good result, but the results will be different relative to the first time we run the experiment
	* Repeating the experiment again with a different combination of training and testing samples, the results are relatively close to the generalization error, but distinct from each other
	* Repeating the process, we get a good approximation of the generalization error, but the precision is poor i.e. all the results are extremely different from one another
	* If we use fewer data points to train the model and more to test the model, the accuracy of the generalization performance will be less but the model will have good precision

* To overcome this problem, we use **cross-validation**
	* Most common out-of-sample evaluation metrics 
	* More effective use of data (each observation is used for both training and testing)
	* In this method, the dataset is split into `K` equal groups. Each group is referred to as a **fold** (i.e., `K-fold`). For example, four folds. Some of the folds can be used as a training set which we use to train the model and the remaining parts are used as a test set, which we use to test the model.
	* For example, we can use three folds for training, then use one fold for testing. This is repeated until each partition is used for both training and testing. At the end, we use the average results as the estimate of out-of-sample error
	* The evaluation metric depends on the model, for example, the `R-squared`
	* The simplest way to apply cross-validation is to call the `cross_val_score()` function of `sklearn`, which performs multiple out-of-sample evaluations
	* The default scoring is `R-Squared`, each element in the array has the average `R-Squared` value in the fold:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr, x_data, y_data, cv=3)

'''
* lr: model type used for cross-validation (here: linear regression)
* x_data: predictive variable data
* y_data: target variable data
* cv: nb of partitions (here: 3 -> data is equally splited into 3 partitions)
* scores: The function returns an array of scores, one for each partition that 
was chosen as the testing set. We can average the result together to estimate 
out of sample R-squared using the mean function NumPi.
'''

np.mean(scores)
```

### Function `cross_val_predict()`

* Return the prediction that was obtained for each element when it was in the test set
* Has a similar interface to `cross_val_score()`
* The input parameters are exactly the same as the cross_val_score function, but the output is a prediction

```python
from sklearn.model_selection import cross_val_score

yhat = cross_val_predict(lr2e, x_data, y_data, cv=3)
```

## 5.2. Overfitting, Underfitting, and Model Selection

* Recall: The goal of `Model Selection` is to **determine the order of the polynomial** to provide the best estimate of the function `y(x)`
* Underfitting: If we try and fit the function with a linear function, the line is not complex enough to fit the data. As a result, there are many errors. This is called underfitting, where the model is too simple to fit the data
* Overfitting: If we increase the order of the polynomial, the model fits better, but the model is still not flexible enough and exhibits underfitting. This is an example of the 8th order polynomial used to fit the data. We see the model does well at fitting the data and estimating the function even at the inflection points. Increasing it to a 16th order polynomial, the model does extremely well at tracking the training point but performs poorly at estimating the function. This is especially apparent where there is little training data. The estimated function oscillates not tracking the function. This is called overfitting, where the model is too flexible and fits the noise rather than the function

In the following figure: (a) Underfitting, (b) Fit, (c) Overfitting

![c6_w5_underfit_overfit](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_underfit_overfit.png)


Plot of the mean square error for the training and testing set of different order polynomials

![c6_w5_poly_order](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_poly_order.png)

* x-axis: represents the order of the polynomial
* y-axis: the mean square error
* The training error decreases with the order of the polynomial
* The test error is a better means of estimating the error of a polynomial
* The error decreases 'til the best order of the polynomial is determined then the error begins to increase
* We select the order that minimizes the test error. In this case, it was `8`. 
* Anything on the left would be considered `underfitting`, anything on the right is `overfitting`
* If we select the best order of the polynomial, we will still have some errors. If you recall the original expression for the training points we see a noise term. This term is one reason for the error. This is because the noise is random and we can't predict it. This is sometimes referred to as an irreducible error. 
* There are other sources of errors as well. For example, our polynomial assumption may be wrong. Our sample points may have come from a different function. 


We can calculate different R-squared values as follows

```python
Rsquared_test = []
order = [1, 2, 3, 4] # To test the model with different polynomial order

for n in order:

	# Create a polynomial feature object 
	pr = PolynomialFeatures(degree=n)

	# Transform the training and test data into a 
	# polynomial using the fit transform method
	x_train_pr 	= pr.fit_transform(x_train[['horsepower']])
	x_test_pr 	= pr.fit_transform(x_test[['horsepower']]) 

	# Fit the regression model using the transform data
	lr.fit(x_train_pr, y_train)

	# Calculate the R-squared using the test data and store it in the array
	Rsquared_test.append(lr.score(x_test_pr, y_test))
```


## 5.3. Ridge Regression
* Prevents overfitting
* Overfitting is serious especially when we have many outliers
* Ridge regression controls the magnitude of these polynomial coefficients by introducing the parameter `alpha`
* `alpha` is a parameter we select before fitting or training the model
* Each row in the following table represents an increasing value of `alpha`
* As `alpha` increases the parameters get smaller. This is most evident for the higher order polynomial features
* `alpha` must be selected carefully
	* If `alpha = 0`, the overfitting is evident
	* If `alpha` is too large, the coefficients will approach `0` -> Underfitting
	

Example:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;1&plus;&space;2x-&space;3x^2-&space;2x^3-&space;12x^4-&space;40x^5&plus;&space;80x^6-&space;141x^8-&space;38x^9&plus;&space;75x^{10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;1&plus;&space;2x-&space;3x^2-&space;2x^3-&space;12x^4-&space;40x^5&plus;&space;80x^6-&space;141x^8-&space;38x^9&plus;&space;75x^{10}" title="\hat{y} = 1+ 2x- 3x^2- 2x^3- 12x^4- 40x^5+ 80x^6- 141x^8- 38x^9+ 75x^{10}" /></a>


`alpha`	| `x` | `x^2` | `x^3` | `x^4` | `x^5` | `x^6` | `x^7` | `x^8` | `x^9` | `x^10`
--------|-----|-------|-------|-------|-------|-------|-------|-------|-------|--------
0 		| 2   | -3    | -2    | -12   | -40   | 80    | 71    | -141  | -38   | 75
0.001	| 2   | -3    | 7     | 5     | 4     | -6    | 4     | -4    | 4     | 6
0.01	| 1   | -2    | -5    | -0.04 | -0.15 | -1    | 1     | -0.5  | 0.3   | 1
1		| 0.5 | -1    | -1    | -0.614| 0.7   | -0.38 | -0.56 | -0.21 | -0.5  | -0.1
10 		| 0   | -0.5  | -0.4  | -0.37 | -0.3  | -0.3  | -0.22 | -0.22 | -0.22 | -0.17

* For `alpha = 0.001`, the overfitting begins to subside
* For `alpha = 0.01`, the estimated function tracks the actual function (good value)
* For `alpha = 1`, we see the first signs of underfitting. The estimated function does not have enough flexibility
* For `alpha = 10`, we see extreme underfitting

![c6_w5_alpha_0](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_alpha_0.png)

![c6_w5_alpha_0p001](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_alpha_0p001.png)

![c6_w5_alpha_0p01](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_alpha_0p01.png)

![c6_w5_alpha_1](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_alpha_1.png)

![c6_w5_alpha_10](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_alpha_10.png)


### Ridge Regression with `sklearn`

```python
from sklearn.linear_model import Ridge

RigeModel = Ridge(alpha=0.1)
RigeModel.fit(X, y)
Yhat = RigeModel.predict(X)
```

Procedure:
Set `alpha` value -> Train -> Predict -> Calculate `R-squared` and keep track of it
-> Set another `alpha` value and repeat the process
Finally: Select the value of `alpha` that maximizes the `R-squared`
Note: Instread of `R-squared`, we can also use other metric like MSE

### Overfitting problem 

* The overfitting problem is even worse if we have lots of features

Example: Plot: `R-Squared` vs. `alpha`

![c6_w5_R2_vs_alpha](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_R2_vs_alpha.png)

* We use several features from our used car data set and a second order polynomial function
* The training data is in red and validation data is in blue
* We see as the value for alpha increases, the value of R-squared increases and converges at approximately `0.75`
* In this case, we select the maximum value of alpha because running the experiment for higher values of `alpha` have little impact
* Conversely, as `alpha` increases, the R-squared on the test data decreases
* This is because the term alpha prevents overfitting
* This may improve the results in the unseen data, but the model has worse performance on the test data.

* Note: `alpha` increases -> `R-squared` on the test data decreases, and vice versa


## 5.4. Grid Search

**Summary**
* Grid Search allows us to scan through multiple free parameters with few lines of code
* Parameters like the alpha term discussed in the previous video are not part of the fitting or training process. These values are called hyperparameters

**Hyperparameters**
* In Ridge Regression, the term `alpha` is called a **hyperparameter**
* `scikit-learn` has a mean of automatically interating over these hyperparameter using cross-validation called **Grid Search**
	* `Grid Search` takes the model or objects you would like to train and different values of the hyperparameters
	* It then calculates the `MSE` or `R-squared` for various hyperparameter values, allowing you to choose the best values
	* Finally we select the hyperparameter that minimizes the error

![c6_w5_grid_search](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_grid_search.png)

**Select hyperparameters**
* Split the dataset into 3 parts: training set, validation set, and test set
	* Train the model with different hyperparameters on the *training set*
	* Use the `R-squared` or `MSE` for each model
	* Select the hyperparameter that minimizes the mean squared error or maximizes the R-squared on the *validation set*
	* Finally test the model performance using the *test data*

![c6_w5_grid_search_hyperparameter](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c6_w5_grid_search_hyperparameter.png)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV # dictionary of parameter values


# Grid Search parameters: using Python dictionary
# each alpha will return a R^2 value
parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}] 

# Ridge regression also has the option to normalize the data
parameters = [{'alpha': [1, 10, 100, 1000], 'normalized': [True, False]}] 

# Create a ridge regression object or model.
RR = Ridge()

# Create a GridSearchCV object. Default scoring method is R-squared
Grid = GridSearchCV(RR, parameters, cv=4) # cv: number of folds

# Fit the object
RR.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# Find the best values for the free parameters
Grid.best_estimator_

# Find the mean score on the validation data
scores = Grid.cv_results_
scores['mean_test_score']

# We can print out the score for different parameter values
for param, mean_val, mean_test in zip(scores['params'], scores['mean_test_score'], scores['mean_train_score']):
	print(param, "R^2 on test data: ", mean_val, "R^2 on train data: ", mean_test)

```




## QUIZ

You train a ridge regression model, you get a R^2 of 1 on your training data and you get a R^2 of 0 on your validation data, what should you do:
* your model is overfitting, increase the parameter alpha

Output of `cross_val_score(lre, x_data, y_data, cv=2)`
* The average R^2 on the test data for each of the two folds

Output of `cross_val_predict (lr2e, x_data, y_data, cv=3)`
* The predicted values of the test data using cross-validation
