<!-- 
To make LaTeX equation and embed to markdown as HTML:	
https://www.codecogs.com/eqnedit.php?latex=\mathcal{W}(A,f)&space;=&space;(T,\bar{f})

To insert figure:
![img_name](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/img_name.png)
 -->



# COURSE 8: MACHINE LEARNING WITH PYTHON

## Week 1: Introduction to Machine Learning


## Week 2: Regression



## Week 3: Classification


### 3.1. K-Nearest Neighbours

#### 3.1.1. Introduction to Classification

* Classification is a **supervised learning** approach
* Categorizing or classifying some unknown items into a discrete set of classes
* Classification attempts to learn the relationship between **a set of feature variables** and **a target variable of interest**
* The target attribute in classification is a categorical variable with discrete values
* Multi-class classifier: A classifier that can predict a field with multiple discrete values, such as `DrugA`, `DrugX` or `DrugY`
* Applications of classification: Email filtering, speech recognition, handwriting recognition, biometric identification, document classification, and so forth
* Classification algorithms in ML:
	* Decision Tree (`ID3`, `C4.5`, `C5.0`)
	* Naïve Bayes
	* Linear Discrimimant Analysis
	* k-Nearest Neighbour (KNN)
	* Logistic Regression
	* Neural Networks
	* Support Vector Machine (SVM)


#### 3.1.2. K-Nearest Neighbours


Example: Build a classifier, using the row `0` to `7` to predict the class of row `8`

![c8_w3_KNN_example_1](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_example_1.png)

For sake of demonstration, let's use only two fields as predictors specifically, age and income, and then, plot the customers based on their group membership:

![c8_w3_KNN_example_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_example_2.png)

* Let's say that we have a new customer, i.e., record number `8`, with a known age and income. How can we find the class of this customer? Can we find one of the closest cases and assign the same class label to our new customer? Can we also say that the class of our new customer is most probably group `4`, i.e, `Total Service`, because it's nearest neighbor is also of class `4`? Yes, we can. In fact, it is the first nearest neighbor.
* **Question**: To what extent can we trust our judgment which is based on the first nearest neighbor? It might be a poor judgment especially, if the first nearest neighbor is **a very specific case** or **an outlier**
* Now, rather than choose the first nearest neighbor, what if we chose the `5` nearest neighbors and did a majority vote among them to define the class of our new customer? 
* In this case, we'd see that `3` out of `5` nearest neighbors tell us to go for class `3`, which is `Plus Service`. This result makes more sense than choosing only one (1st) nearest neighbor
* This example highlights the intuition behind the `K-Nearest Neighbors` (`KNN`) algorithm, with `k = 5`


![c8_w3_KNN_example_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_example_3.png)

##### Definition:

* `KNN` is a classification algorithm that takes a bunch of labeled points and uses them to learn how to label other points
* `KNN` classifies cases based on their similarity to other cases
* In `KNN`, data points that are near each other are said to be neighbors
* `KNN` is based on this paradigm: Similar cases with the same class labels are near each other. Thus, the distance between two cases is a measure of their dissimilarity
* There are different ways to calculate the similarity or conversely, the distance or dissimilarity of two data points. For example, this can be done using **Euclidean distance** (a specific type of the **Minkowski distance**). The dissimilarity measure method is highly dependent on datatype and also the domain that classification is done for

#### `KNN` Procedure:

(1) Pick a value for `K`
(2) Calculate the distance of unknown case from all cases
(3) Select the `K`-observations in the training data that are "nearest" to the unknown data point
(4) Predict the response of the unknown data point using the most popular response value from the `K` nearest neighbors

For (2), one can use the **Euclidean distance** to calculate the similarity/distance between 2 customers (1-dimensional space)

<!-- 
`dis(x_1, x_2) = \sqrt{\sum_{i=0}^{n} (x_{1i} - x_{2i})^2}` 
-->

<a href="https://www.codecogs.com/eqnedit.php?latex=dis(x_1,&space;x_2)&space;=&space;\sqrt{\sum_{i=0}^{n}&space;(x_{1i}&space;-&space;x_{2i})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dis(x_1,&space;x_2)&space;=&space;\sqrt{\sum_{i=0}^{n}&space;(x_{1i}&space;-&space;x_{2i})^2}" title="dis(x_1, x_2) = \sqrt{\sum_{i=0}^{n} (x_{1i} - x_{2i})^2}" /></a>

![c8_w3_KNN_1D_distance](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_1D_distance.png)

If we have more than one feature: We can reuse the Euclidean function in a multi-dimensional space.

For example: 2D-space:

![c8_w3_KNN_2D_distance](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_2D_distance.png)

Or 3D-space:

![c8_w3_KNN_3D_distance](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_KNN_3D_distance.png)

**Question**: How do we choose the right value of `K`?
* `K = 1`: A low value of K causes a highly complex model as well, which might result in **overfitting** of the model
* `K = 20`: The model becomes overly generalized
* General solution: 
	* Reserve a part of your data for testing the accuracy of the model
	* Once you've done so, choose `K = 1` and then use the training part for modeling and calculate the accuracy of prediction using all samples in your test set
	* Repeat this process increasing the `K` and see which `K` is best for your model

##### Computing continuous targets using `KNN`
* `KNN` can also be used for regression
* In this situation, the **average or median target value of the nearest neighbors** is used to obtain the predicted value for the new case
* For example, assume that you are predicting the price of a home based on its feature set, such as number of rooms, square footage, the year it was built, and so on. You can easily find the three nearest neighbor houses of course not only based on distance but also based on all the attributes and then predict the price of the house as the medium of neighbors


#### 3.1.3. Evaluation Metrics in Classification: `Jaccard index`, `F1-score`, and `Log Loss`


##### `Jaccard index` (also `Jaccard similarity coefficient`)

* `y`: actual labels
* `yhat`: predicted labels

The Jaccard index is given by:


<!-- 
	J(y, \hat{y}) = \cfrac{y \cap \hat{y}}{y \cup \hat{y}}= \cfrac{y \cap \hat{y}}{|y|+ |\hat{y}|- |y \cup \hat{y}|} 
-->


<a href="https://www.codecogs.com/eqnedit.php?latex=J(y,&space;\hat{y})&space;=&space;\cfrac{y&space;\cap&space;\hat{y}}{y&space;\cup&space;\hat{y}}=&space;\cfrac{y&space;\cap&space;\hat{y}}{|y|&plus;&space;|\hat{y}|-&space;|y&space;\cup&space;\hat{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(y,&space;\hat{y})&space;=&space;\cfrac{y&space;\cap&space;\hat{y}}{y&space;\cup&space;\hat{y}}=&space;\cfrac{y&space;\cap&space;\hat{y}}{|y|&plus;&space;|\hat{y}|-&space;|y&space;\cup&space;\hat{y}|}" title="J(y, \hat{y}) = \cfrac{y \cap \hat{y}}{y \cup \hat{y}}= \cfrac{y \cap \hat{y}}{|y|+ |\hat{y}|- |y \cup \hat{y}|}" /></a>


![c8_w3_Jaccard_index](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Jaccard_index.png)

Example:
```python
y 		= [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
yhat 	= [1, 1, 0, 0, 0, 1, 1, 1, 1, 1]

J(y,yhat) = 8/(10 + 10 - 8)  # Jaccard index = 0.66

```


##### `F1-score` (derived from the `confusion matrix`)

Take a look at the `confusion matrix` below:

![c8_w3_Confusion_matrix](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Confusion_matrix.png)

* For true value `churn = 1`, there are `6/15` correctly predicted
* For true value `churn = 0`, there are `24/25` correctly predicted (only `1` false) -> So it has done a good job in predicting the customers with a churn value of `0` 

* A good thing about the confusion matrix is that it shows the model's ability to correctly predict or separate the classes
* In the specific case of a binary classifier, such as this example, we can interpret these numbers as the count of true positives (`TP`), false positives (`FP`), true negatives (`TN`), and false negatives (`FN`)

![c8_w3_Confusion_matrix_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Confusion_matrix_2.png)


Based on the count of each section, we can calculate the precision and recall of each label:
* `Precision` (measure of accuracy): `Precision = TP/(TP + FP)`
* `Recall` (true positive rate): `Recall = TP/(TP + FN)` 
* `F1-score` (harmonic average of the precision and recall): `F1-score = 2*(Precision*Recall)/(Precision + Recall)`
	* `F1-score` ranges in `[0,1]`, `F1-score = 1` represents perfect precision and recall

For the above example, we have:

			| precision | recall | f1-score
------------|-----------|--------|-----------
churn = 0	| 0.73		| 0.96	 | 0.83
churn = 1	| 0.86		| 0.40	 | 0.55

The average accuracy = average `F1-score` = 0.72

**NOTE**: Both `Jaccard index` and `F1-score` can be used for multi-class classifier as well.


![c8_w3_Confusion_matrix_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Confusion_matrix_3.png)


##### `Log-loss` 

* Sometimes the output of a classifier is the **probability** of a class label, instead of the label
* For example, in logistic regression, the output can be the probability of customer churn, i.e., `yes` (or equals to `1`). This probability is a value between `0` and `1`
* Logarithmic loss (also known as `Log loss`) measures the performance of a classifier where the predicted output is a probability value between `0` and `1`
* So, for example, predicting a probability of `0.13` when the actual label is `1`, would be bad and would result in a **high log loss**
* We can calculate the log loss for each row using the log loss equation, which measures how far each prediction is, from the actual label

<!-- 
LogLossForEachRow = ylog(\hat{y}) + (1-y)log(1 - \hat{y})
-->

<a href="https://www.codecogs.com/eqnedit.php?latex=LogLoss&space;=&space;ylog(\hat{y})&space;&plus;&space;(1-y)log(1&space;-&space;\hat{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LogLoss&space;=&space;ylog(\hat{y})&space;&plus;&space;(1-y)log(1&space;-&space;\hat{y})" title="LogLoss = ylog(\hat{y}) + (1-y)log(1 - \hat{y})" /></a>


![c8_w3_Log_Loss](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Log_Loss.png)


* Then, we calculate the average log loss across all rows of the test set

<!-- 
LogLoss = -\cfrac{1}{n} \sum \left [ ylog(\hat{y}) + (1-y)log(1 - \hat{y}) \right ]
-->

<a href="https://www.codecogs.com/eqnedit.php?latex=LogLoss&space;=&space;-\cfrac{1}{n}&space;\sum&space;\left&space;[&space;ylog(\hat{y})&space;&plus;&space;(1-y)log(1&space;-&space;\hat{y})&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LogLoss&space;=&space;-\cfrac{1}{n}&space;\sum&space;\left&space;[&space;ylog(\hat{y})&space;&plus;&space;(1-y)log(1&space;-&space;\hat{y})&space;\right&space;]" title="LogLoss = -\cfrac{1}{n} \sum \left [ ylog(\hat{y}) + (1-y)log(1 - \hat{y}) \right ]" /></a>

* It is obvious that more ideal classifiers have progressively smaller values of log loss
* So, the classifier with **lower log loss** has **better accuracy**

![c8_w3_Log_Loss_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Log_Loss_2.png)

**Summary**: An ideal classifier must have:
* `F1-score` close to `1`
* `Jaccard index` close to `1`
* `Log-loss` close to `0`


#### Lab: KNN

### 3.2. Decision Trees

#### 3.2.1. Introduction to Decision Trees

Example:
![c8_w3_Decision_Tree](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Decision_Tree.png)

Building Decision Tree:
* Decision trees are built by splitting the training set into distinct nodes, where one node contains all of or most of one category of the data.
* Decision trees are about testing an attribute and branching the cases based on the result of the test
	* Each internal node corresponds to a test
	* Each branch corresponds to a result of the test
	* Each leaf node assigns a patient to a class

![c8_w3_Decision_Tree_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Decision_Tree_2.png)

Decision Tree learning algorithm:
(1) Choose an attribute from your dataset
(2) Calculate the significance of attribute in splitting of data
(3) Split data based on the value of the best attribute
(4) Go to Step (1)

#### 3.2.2. Building Decision Trees

![c8_w3_Decision_Tree_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Decision_Tree_3.png)

**Decision trees:** 
* Are built using recursive partitioning to classify the data
* What is important in making a decision tree, is to determine which attribute is the best or more predictive to split data based on the feature
* The choice of attribute to split data is very important and it is all about purity of the leaves after the split
* A node in the tree is considered pure if in 100% of the cases, the nodes fall into a specific category of the target field
* In fact, the method uses recursive partitioning to split the trading records into segments by minimizing the impurity at each step
* Impurity of nodes is calculated by entropy of data in the node. 

**Entropy:**

<!-- 
Entropy = -p(A)\log(A) - p(B)\log(B)
 -->

<a href="https://www.codecogs.com/eqnedit.php?latex=Entropy&space;=&space;-p(A)\log(A)&space;-&space;p(B)\log(B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Entropy&space;=&space;-p(A)\log(A)&space;-&space;p(B)\log(B)" title="Entropy = -p(A)\log(A) - p(B)\log(B)" /></a>

* Is the amount of information disorder or the amount of randomness  (or uncertainty) in the data
	* The entropy in the node depends on how much random data is in that node and is calculated for each node
* In decision trees, we're looking for trees that have the smallest entropy in their nodes
	* The lower the entropy, the less uniform the distribution, the purer the node
	* The entropy is used to calculate the homogeneity of the samples in that node
* If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one. 

![c8_w3_Entropy](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Entropy.png)

Example: 

Entropy of the data before splitting it:

![c8_w3_Entropy_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Entropy_2.png)

Entropy after splitting:

* If one chooses `Cholesterol` 

![c8_w3_Entropy_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Entropy_3.png)

* If one chooses `Sex` 

![c8_w3_Entropy_4](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Entropy_4.png)

* Now, which attribute is the best between `Cholesterol` and `Sex`?

![c8_w3_Information_Gain](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Information_Gain.png)

Answer: The tree with the higher **information gain** after splitting

**Information gain:** 

* `Information gain = (entropy before split) - (weighted entropy after split)`
* Is the information that can increase the level of certainty after splitting
* Is the entropy of a tree before the split minus the weighted entropy after the split by an attribute
* We can think of information gain and entropy as opposites
	* As entropy or the amount of randomness decreases, the information gain or amount of certainty increases and vice versa
So, constructing a decision tree is all about finding attributes that return the highest information gain

![c8_w3_Information_Gain_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Information_Gain_2.png)

So, the `Sex` attribute, which gives the higher information gain, is the best. So, we select the sex attribute as the first splitter
* We repeat the process for each branch and test each of the other attributes to continue to reach the most pure leaves. With this, we can find the next attribute after branching by the `Sex` attribute. This is the way we build a decision tree.

![c8_w3_Information_Gain_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Information_Gain_3.png)

### 3.3. Logistic Regression

#### 3.3.1. Introduction to Logistic Regression

**Definition:**
* Logistic regression is a statistical and machine learning technique for classifying records of a dataset based on the values of the input fields
* In short: LR is a classification algorithm for categorical variables

![c8_w3_Logistic_Regression](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression.png)

In logistic regression, we use one or more **independent variables** such as `tenure`, `age` and `income` to predict an outcome, such as `churn`, which we call the **dependent variable** representing whether or not customers will stop using the service

![c8_w3_Logistic_Regression_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression_2.png)


* Logistic regression is analogous to linear regression, but tries to predict a **categorical** or **discrete target field** instead of a **numeric** one
	* Linear regression: Predict a **continuous value** of variables such as the price of a house, blood pressure of a patient, or fuel consumption of a car
	* Logistic regression: Predict a **binary value** such as `yes/no`, `true/false`, `successful/not successful`, `pregnant/not pregnant`, and so on, all of which can be coded as `0` or `1`
		* In logistic regression **dependent variables** should be **continuous**. If **categorical**, they should be **dummy** or **indicator coded**. This means we have to transform them to some continuous value. 
* Note that logistic regression can be used for BOTH **binary classification** (binary logistic regression) and **multi-class classification** (multinomial logistic regression)

**Example:**
* Predict the probability of a person having a heart attack within a specified time period, based on our knowledge of the person's age, sex and body mass index
* Predict the chance of mortality in injured patients or to predict whether a patient has a given disease such as diabetes based on observed characteristics of that patient such as weight, height, blood pressure and results of various blood tests and so on
* In marketing: Predict the likelihood of a customer purchasing a product or halting a subscription as we've done in our churn example
* Predict the probability of failure of a given process, system or product
* Predict the likelihood of a homeowner defaulting on a mortgage


**When logistic regression is useful:**
(1) If the data is binary 
(2) If we need preobabilistic results
	* In fact, logistic regression predicts the probability of that sample and we map the cases to a discrete class based on that probability
(3) If we need a linear decision boundary. 
	* If your data is linearly separable. The decision boundary of logistic regression is a line or a plane or a hyper plane
	* A classifier will classify all the points on one side of the decision boundary as belonging to one class, and all those on the other side as belonging to the other class
	* Note that in using logistic regression, we can also achieve a complex decision boundary using polynomial processing as well

![c8_w3_Logistic_Regression_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression_3.png)

(4) If we need to understand the impact of a feature


#### 3.3.2. Logistic Regression vs. Linear Regression

Linear regression: Use linear function: `theta^T * X`
	* If we use the regression line to calculate the class of a point, it always returns a number such as three or negative two, and so on. Then, we should use a threshold, for example, `0.5`, to assign that point to either class of zero or one. This threshold works as a step function that outputs zero or one regardless of how big or small, positive or negative the input is.

Logistic regression: Use logistic function (sigmoid function): `sigma(theta^T * X)`
	* Give us the probability of falling in the class

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma(\theta^{\top}X)&space;=&space;\frac{1}{1&plus;\exp(-\theta^{\top}X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma(\theta^{\top}X)&space;=&space;\frac{1}{1&plus;\exp(-\theta^{\top}X)}" title="\sigma(\theta^{\top}X) = \frac{1}{1+\exp(-\theta^{\top}X)}" /></a>

**Comment**: Depicting on the in sigmoid plot, when Theta transpose `x` gets bigger, the value of the sigmoid function gets closer to `1`. Also, if the Theta transpose `x` is very small, the sigmoid function gets closer to `0`. So, the sigmoid functions output is always between `0` and `1`, which makes it proper to interpret the results as probabilities

![c8_w3_Logistic_Regression_sigmoid](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression_sigmoid.png)

**Summary**:
Logistic regression: 
* Logistic model: `Y^hat = P(Y=1|X)`: calculate the probability of falling into class Y
* Function: `sigmoid`: `Y^hat = sigmoid(X) = sigma(theta^T * X) =  P(Y=1|X)`
* Input: Independent variables `X`
* Output: Dependent variable `Y` (the class we want to predict)
* The value of vector `Theta` can be found through the training process
Example: `P(churn=1|income,age) = 0.8` -> `P(churn=0|income,age) = 0.2`

**Training process to find `Theta`**:
1. Initialize `Theta` vector with random values, e.g., `theta = [-1,2]`  
2. Calculate `y^hat = sigma(theta^T * X)` for a customer, e.g., `y^hat = sigma([-1,2]*[2,5]) = 0.7`
3. Compare the output of `y^hat` with the actual ouput of customer, `y`, and record it as error: `Error = 1-0.7 = 0.3`
4. Calculate error for all customer. The total error is the cost of the model and is given by the cost function: `cost = J(theta)`
5. Change `theta` to reduce the cost
6. Go back to Step 2, until the cost is low enough

**Two questions**: 
1. How to change `theta to reduce the cost? -> Many way, the most popular one is using `gradient descent`
2. When should we stop the iterations? -> By calculating the accuracy of the model and stop it when it's satisfactory


#### 3.3.3. Logistic Regression Training

Cost function: represents the difference between the predicted value (`y^hat = sigma(theta^T*X)`) and the actual value (`y`):

<a href="https://www.codecogs.com/eqnedit.php?latex=cost(\hat{y},&space;y)&space;=&space;\frac{1}{2}(\sigma(\theta^{\top}X)&space;-&space;y)^2&space;\\&space;\text{Mean&space;Squared&space;Error&space;(MSE):}~J(\theta)&space;=&space;\frac{1}{m}&space;\sum_{i-1}^{m}&space;cost(\hat{y},&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?cost(\hat{y},&space;y)&space;=&space;\frac{1}{2}(\sigma(\theta^{\top}X)&space;-&space;y)^2&space;\\&space;\text{Mean&space;Squared&space;Error&space;(MSE):}~J(\theta)&space;=&space;\frac{1}{m}&space;\sum_{i-1}^{m}&space;cost(\hat{y},&space;y)" title="cost(\hat{y}, y) = \frac{1}{2}(\sigma(\theta^{\top}X) - y)^2 \\ \text{Mean Squared Error (MSE):}~J(\theta) = \frac{1}{m} \sum_{i-1}^{m} cost(\hat{y}, y)" /></a>


**Using Gradient Descent to minimize the cost**

* `mu`: learning rate

![c8_w3_Logistic_Regression_gradient_descent](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression_gradient_descent.png)

**Training process recap**

![c8_w3_Logistic_Regression_training](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c8_w3_Logistic_Regression_training.png)





### 3.4. Support Vector Machine

* A Support Vector Machine is a supervised algorithm that can classify cases by finding a **separator**
	* First: Mapping data to a **high dimensional** feature space so that data points can be categorized, even when the data are not otherwise linearly separable. 
	* Then: A separator is estimated for the data. The data should be transformed in such a way that a separator could be drawn as a hyperplane.

Kernelling: Mapping data into a higher dimensional space, in such a way that can change a linearly inseparable dataset into a linearly separable dataset.


Pros:
* Accurate in high-dimensional spaces
* Memory efficient
Cons:
* Prone to over-fitting
* No probability estimation
* Small datasets

Applications:
* Image recognition
* Text category assignment
* Spam detection
* Sentiment analysis
* Gene expression classification
* Regression, outlier detection and clustering



### Quiz:

Which one IS NOT a sample of classification problem?
	To predict the category to which a customer belongs to.
	To predict whether a customer switches to another provider/brand.
	*To predict the amount of money a customer will spend in one year.*
	To predict whether a customer responds to a particular advertising campaign or not.

Which of the following sentences are TRUE about Logistic Regression (select all the options that are correct)?? ALL
	Logistic regression can be used both for binary classification and multi-class classification
	Logistic regression is analogous to linear regression but takes a categorical/discrete target field instead of a numeric one.
	In logistic regression, the dependent variable is binary.


Which of the following examples is/are a sample application of Logistic Regression (select all the options that are correct)?
	*The probability that a person has a heart attack within a specified time period using person's age and sex.*
	*Customer's propensity to purchase a product or halt a subscription in marketing applications.*
	*Likelihood of a homeowner defaulting on a mortgage.*
	Estimating the blood pressure of a patient based on her symptoms and biographical data.

Which one is TRUE about kNN algorithm?
	kNN is a classification algorithm that takes a bunch of unlabelled points and uses them to learn how to label other points.
	*kNN algorithm can be used to estimate values for a continuous target.*

What is "information gain" in decision trees??
	It is the information that can decrease the level of certainty after splitting in each node.
	*It is entropy of a tree before split minus weighted entropy after split by an attribute.*
	It is the amount of information disorder, or the amount of randomness in each node.




## Week 4: Clustering

### 4.1. Introduction to Clustering

Why clustering?
* Exploratory data analysis
* Summary generation
* Outlier detection
* Finding duplicates
* Pre-processing step

Clustering algorithms
* Partitioned-based clustering
	* Relatively efficient
	* Example: `k-means`, `k-median`, `fuzzy c-means`
* Hierarchical clustering
	* Produces trees of clusters
	* Example: `agglomerative`, `divisive`
* Density-based clustering
	* Produces arbitrary shaped clusters
	* Example: `DBSCAN`


Applications:
* Retail/Marketing:
	* Identify buying patterns of customers
	* Recommending new books or movies to new customers
* Banking: 
	* Fraud detection in credit card use
	* Identify clusters of customers (e.g., loyal)
* Insurance:
	* Fraud detection in claim analysis
	* Insurance risk of customers
* Publication:
	* Auto-categorizing news based on their content
	* Recommending similar news articles 
* Medicine:
	* Characterizing patient behavior
* Biology:
	* Clustering genetic markers to identify family ties



### 4.2. `k-means` Clustering

**Objective**
	* To form clusters in such a way that similar samples go into a cluster, and dissimilar samples fall into different clusters.
	* To minimize the “intra cluster” distances and maximize the “inter-cluster” distances.
	* To divide the data into non-overlapping clusters without any cluster-internal structure

**Note**
* As k-means is an iterative algorithm, there is no guarantee that it will converge to the global optimum, and the result may depend on the initial clusters
* Tt means, this algorithm is guaranteed to converge to a result but the result may be a local optimum, i.e., not necessarily the best possible outcome.
* To solve this problem, it is common to run the whole process multiple times with different starting conditions. This means with randomized starting centroids, it may give a better outcome. As the algorithm is usually very fast, it wouldn't be any problem to run it multiple times.

**Mechanism**
1. Randomly placing `k` centroids, one for each cluster
2. Calculate the distance of each point from each centroid
3. Assign each data point (object) to its closest centroid, creating a cluster
4. Recalculate the position of the `k` centroids

**`k-means` Accuracy**
* Extenal approach
	* Compare the clusters with the ground truth, if it is available
	* However, because `k-means` is an unsupervised algorithm, we usually don't have ground truth in real world problems to be used
* Internal approach
	* Average the distance between data points within a cluster

To select the value of `k`: `Elbow` method
* The `elbow point` (value of `k`) is where the rate of accuracy decreases sharply


**`k-means` Clustering Recap**
* For medium and large-sized databases (relatively efficient)
* Produces sphere-like clusters
* Needs number of clusters (`k`)



### 4.3. Density-Based Spatial Clustering of Applications with Noise (`DBSCAN`)

**Idea**: 
* Visit all the points in the dataset and label them as either core, border, or outlier
* Next, connect core points that are neighbors and put them in the same cluster
* So, a cluster is formed as at least one core point plus all reachable core points plus all their borders

**Characteristics**
* One of the most common clustering algorithm
* Works based on density of objects and two parameters:
	* `R` (radius of neighborhood): of if it includes enough number of points within, we call it a dense area
	* `M` (min number of neighbors): min number of points we want in a neighborhood to define a cluster
* Each point in the dataset is eigther: a `core`, `border`, or `outlier` point
* While `k-means` assigns all points to a cluster even if they do not beling in any cluster, the `DBSCAN`
	* locates regions of high density that are separated from one another by regions of low density
	* separates outliers
* Density-based clustering algorithms are proper for arbitrary shape clusters.

**`DBSCAN` Clustering Recap**
* `DBSCAN` can find arbitrarily shaped clusters
* It can even find a cluster completely surrounded by a different cluster
* `DBSCAN` has a notion of noise and is robust to outliers 
* On top of that, `DBSCAN` makes it very practical for use in many real-world problems because it does not require one to specify the number of clusters such as `k` in `k-means`.




### 4.4. Quiz:
1. Which one is NOT TRUE about k-means clustering?
	k-means divides the data into non-overlapping clusters without any cluster-internal structure.
	The objective of k-means, is to form clusters in such a way that similar samples go into a cluster, and dissimilar samples fall into different clusters.
	*As k-means is an iterative algorithm, it guarantees that it will always converge to the global optimum.*


2. Which ones are the characteristics of DBSCAN? ALL
	DBSCAN can find arbitrarily shaped clusters
	DBSCAN can find a cluster completely surrounded by a different cluster.
	DBSCAN has a notion of noise, and is robust to outliers
	DBSCAN does not require one to specify the number of clusters such as k in k-means

3. Which of the following is an application of clustering?
	Customer churn prediction
	Price estimation
	*Customer segmentation*
	Sales prediction

4. Which approach can be used to calculate dissimilarity of objects in clustering?
	Minkowski distance
	Euclidian distance
	Cosine similarity
	*All of the above*

5. How a center point (centroid) is picked for each cluster in k-means?
	*We can randomly choose some observations out of the data set and use these observations as the initial means.*
	*We can create some random points as centroids of the clusters.*
	We can select it through correlation analysis.



## Week 5: Recommender Systems

### 5.1. Intro to Recommender System
Recommender systems capture the pattern of people's behavior and use it **to predict what else they might want or like**

**Applications**
* What to buy: E-commerce, books, movies -> Netflix, Amazon
* Where to eat 
* What job to apply to
* Who you should be friends with? LinkedIn, Facebook
* Personalize your experience on the web: new platforms, new personalization

**Advantages**
* Broader exposure
* Possibility of continual usage or purchase of products
* Provides better experience

**Two types of RS**
In terms of the type of statement that a consumer might make:
	1. Content-Based:
		* Content-based systems try to figure out what a user's favorite aspects of an item are, and then make recommendations on items that share those aspects. 
		* The main paradigm of a Content-based recommendation system is driven by the statement: “Show me more of the same of what I've liked before." 
	2. Collaborative Filtering
		* Collaborative filtering techniques find similar groups of users, and provide recommendations based on similar tastes within that group. In short, it assumes that a user might be interested in what similar users are interested in.
		* Collaborative filtering is based on a user saying, “Tell me what's popular among my neighbors because I might like it too.” 	
	* Also, there are Hybrid recommender systems, which combine various mechanisms.
In terms of implementing recommender systems, there are 2 types: 
	1. Memory-based 
		* Use the entire user-item dataset to generate a recommendation system
		* Use statistical techniques to approximate users or items
		* Examples: Pearson Correlation, Cosine Similarity and Euclidean Distance, etc.
	2. Model-based
		* Develop a model of users in an attempt to learn their preferences
		* Models can be created using Machine Learning techniques like regression, clustering, classification, etc.


### 5.2. Content-Based Recommender System

Content-based recommendation system tries to recommend items to the users based on their profile.

Shortcoming (drawbacj) of Content-Based RS:
* Users will only get recommendations related to their preferences in their profile, and recommender engine may never recommend any item with other characteristics -> This problem can be solved by other types of recommender systems such as collaborative filtering

### 5.3. Collaborative Filtering

* User-based collaborative filtering
* Item-based collaborative filtering

Cold start: Refers to the difficulty the recommendation system has when 
	* When there is a new user, and as such a profile doesn't exist for them yet
	* When we have a new item which has not received a rating. Scalability can become an issue as well. 

As the number of users or items increases and the amount of data expands, collaborative filtering algorithms will begin to suffer drops in performance, simply due to growth and the similarity computation.


### 5.4. Quiz

1. What is/are the advantage/s of Recommender Systems ?
* Recommender Systems provide a better experience for the users by giving them a broader exposure to many different products they might be interested in.
* Recommender Systems encourage users towards continual usage or purchase of their product
* Recommender Systems benefit the service provider by increasing potential revenue and better security for its consumers.
**All of the above**

2. What is a content-based recommendation system?
**Content-based recommendation system tries to recommend items to the users based on their profile built upon their preferences and taste.**
* Content-based recommendation system tries to recommend items based on similarity among items.
* Content-based recommendation system tries to recommend items based on the similarity of users when buying, watching, or enjoying something.
* All of above.

3. What is the meaning of "Cold start" in collaborative filtering?
* The difficulty in recommendation when we do not have enough ratings in the user-item dataset.
**The difficulty in recommendation when we have new user, and we cannot make a profile for him, or when we have a new item, which has not got any rating yet.**
* The difficulty in recommendation when the number of users or items increases and the amount of data expands, so algorithms will begin to suffer drops in performance.

4. What is a "Memory-based" recommender system?
* In memory based approach, a model of users is developed in attempt to learn their preferences.
* In memory based approach, we use the entire user-item dataset to generate a recommendation system.
* In memory based approach, a recommender system is created using machine learning techniques such as regression, clustering, classification, etc.

5. What is the shortcoming of content-based recommender systems?
* As it is based on similarity among items and users, it is not easy to find the neighbour users.
* It needs to find similar group of users, so suffers from drops in performance, simply due to growth in the similarity computation.
**Users will only get recommendations related to their preferences in their profile, and recommender engine may never recommend any item with other characteristics**




## Week 6: Final Project


Classification
Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model You should use the following algorithm:

* K Nearest Neighbor(KNN)
* Decision Tree
* Support Vector Machine
* Logistic Regression

**Notice:**
* You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
* You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
* You should include the code of the algorithm in the following cells.
