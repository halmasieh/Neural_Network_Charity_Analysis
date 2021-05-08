# Neural_Network_Charity_Analysis

## Project Overview
A neural network is a powerful machine learning technique that is modeled after neurons in the brain.
Neural networks can drive the performance of the robust and statistical algorithms without having worry about any statistical theory. Because of that, neural network is one of the high demand skills for any data scientist. We explore and implement neural networks using the TensorFlow platform in Python and consider the effectiveness of the neural network for the dataset. Finally, we store and retrieve trained models for more robust uses.
However, we spent most of our time for preparing data. 

Infact, we create a deep learning neural network to analyze and classify the success of charitable donations which can be done as the following steps: 
 - Prepare input data
 - Create Deep Learning models
 - Design, train, evaluate, and export neural networks
 

## Resources
Software: [Jupyter Notebook](https://www.anaconda.com/products/individual)

library: Pandas, Sklearn and Tensorflow

## Results
We create and optimize a deep learning model to analyze and classify the success of charitable donations. So we address the following questions:  

### Data Preprocessing
1- What variable(s) are considered the target(s) for your model?

The taget variable is the values of the the array with the name "IS_SUCCESSFUL". 



<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/y%20variable.PNG"/>





2- What variable(s) are considered to be the features for your model?

We consider all the columns after droping the taget variable "IS_SUCCESSFUL". 



<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Features.PNG"/>





3- What variable(s) are neither targets nor features, and should be removed from the input data?

We dropped the non-beneficial ID columns (variables) "EIN" and "NAME" and these variables are neither targets nor features.



<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Droped%20columns.PNG" width="500" height="500"/>





### Compiling, Training, and Evaluating the Model
1- How many neurons, layers, and activation functions did you select for your neural network model, and why?

As shown in the following: 



<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Model%201.PNG" width="400" height="500"/>




We have considered 2 dense layer with ( neurons=80 , activation=tanh) and ( neurons=30 , activation=tanh), repectively. 
In fact, according to researchers on deep learning, chooing three layers would be  good enough for the most of neural network models. The number of neurons is predetermined for each layer. The activation functions are selected in order to obtain higher aacuracy model. In the output layer, the sigmoid function has a characteristic “S” — shaped curve which transforms the output between the range 0 and 1 and it would be effective for the probability obtained in the binary classification.

2- Were you able to achieve the target model performance?

The loss and accuracy for the model are obtaind as follow:




<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/loss-accuracy-model1.PNG"/>





It seems that the model does not have a high accuracy. It is very likely that by manipulating the dataset and changing the number of neurons, layers and activation function, we may achieve a higher accuracy.

3- What steps did you take to try and increase model performance?

In order to increase the accuracy of the model, we perform the following steps in the AlphabetSoupCharity - Optimization:

- Keep the feature "NAME" and  add a bin "other" for the value counts less than 5 words
- Increase the the number of values for the created bin of the value counts APPLICATION_TYPE 
- Use different activation function in the first and second hidden layer
- Change the number of neurons in the the first and second layer and add the third layer with 20 neurons as shown below:







<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Model%202.PNG" width="500" height="500"/>







Therefore, the model is optimized, and the predictive accuracy is increased to over 75% as below:




<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/loss-accuracy-model2.PNG"/>





## Summary

We summarize this analysis as below:

1- We preprocessed the dataset in order to compile, train, and evaluate the neural network model, using Pandas and the Scikit-Learn’s StandardScaler(),  

2- We designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization would be successful based on the features in the dataset, using TensorFlow. 

3- We needed to think about how many inputs there are before determining the number of neurons and layers in our model. 
Once we have completed that step, then we compiled, trained, and evaluated our binary classification model to calculate the model’s loss and accuracy.

4- We optimized our model in order to achieve a target predictive accuracy higher than 75% and that goal was achieved.

### Logistic Regression VS Neural Network
There are some factors to consider when selecting a model for our dataset. First, neural networks are prone to overfitting and can be more difficult to train than a straightforward logistic regression model. Therefore, if we are trying to build a classifier with limited data points (typically fewer than a thousand data points), or if our dataset has only a few features, neural networks may be overcomplicated. Additionally, logistic regression models are easier to dissect and interpret than their neural network counterparts, which tends to put more traditional data scientists and non-data experts at ease. In contrast, neural networks (and especially deep neural networks) thrive in large datasets. Datasets with thousands of data points, or datasets with complex features, may overwhelm the logistic regression model, while a deep learning model can evaluate every interaction within and across neurons. Therefore, the decision between using a logistic regression model and basic neural network model is nuanced and, in most cases, a matter of preference for the data scientist.

### SVM VS Deep Learning
SVMs are a type of binary classifier that use geometric boundaries to distinguish data points from two separate groups. More specifically, SVMs try to calculate a geometric hyperplane that maximizes the distance between the closest data point of both groups.

Unlike logistic regression, which excels in classifying data that is linearly separable but fails in nonlinear relationships, SVMs can build adequate models with linear or nonlinear data. Due to SVMs' ability to create multidimensional borders, SVMs lose their interpretability and behave more like the black box machine learning models, such as basic neural networks and deep learning models.

SVMs, like neural networks, can analyze and interpret multiple data types, such as images, natural language voice and text, or tabular data. SVMs perform one task and one task very well—they classify and create regression using two groups. In contrast, neural networks and deep learning models are capable of producing many outputs, which means neural network models can be used to classify multiple groups within the same model. Over the years, techniques have been developed to create multiple SVM models side-by-side for multiple classification problems, such as creating multiple SVM kernels. However, a single SVM is not capable of the same outputs as a single neural network.

If we only compare binary classification problems, SVMs have an advantage over neural network and deep learning models. Neural networks and deep learning models will often converge on a local minima. In other words, these models will often focus on a specific trend in the data and could miss the "bigger picture." SVMs are less prone to overfitting because they are trying to maximize the distance, rather than encompass all data within a boundary. Despite these advantages, SVMs are limited in their potential and can still miss critical features and high-dimensionality relationships that a well-trained deep learning model could find. However, in many straightforward binary classification problems, SVMs will outperform the basic neural network, and even deep learning models with ease.

