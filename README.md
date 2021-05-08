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

As shown in the following 



<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Model%201.PNG" width="500" height="500"/>




We have considered 2 dense layer with ( neurons=80 , activation=tanh) and ( neurons=30 , activation=tanh), repectively. 
In fact, based on researchers on deep learning, chooing three layers would be  good enough for the most of neural network models. The number of neurons is predetermined for each layer. The activation functions are selected in order to obtain higher aacuracy model. In the output layer, the sigmoid function has a characteristic “S” — shaped curve which transforms the output between the range 0 and 1 an dit would be effective for the probability obtained in the binary classification.

2- Were you able to achieve the target model performance?

The loss and accuracy for the model are obtaind as follow:




<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/loss-accuracy-model1.PNG"/>





It seems that the model does not have a high accuracy. It is very likely that by manipulating the data set and changing the number of neurons, layers and activation function, we may achieve a higher accuracy.

3- What steps did you take to try and increase model performance?
In order to increase the accuracy of the model, we perform the following steps in the AlphabetSoupCharity - Optimization:

- Keep the feature "NAME" and  add a bin "other" for the value counts less than 5 words
- Increase the the number of values for the created bin of the value counts APPLICATION_TYPE 
- Use different activation function in the first and second hidden layer
- Change the number of neurons in the the first and second layer and add the third layer with 20 neurons as shown below:




<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/Model%202.PNG" width="500" height="500"/>






- Keep the number of epochs to the training regimen 

Therefore, the model is optimized, and the predictive accuracy is increased to over 75% as below:




<img src="https://github.com/halmasieh/Neural_Network_Charity_Analysis/blob/main/Resources/loss-accuracy-model2.PNG"/>





## Summary









We summarize this analysis as below:

1- Preprocess the dataset in order to perform PCA in step 2, using Pandas.

2- Reduce the dimensions of the X DataFrame to three principal components and place these dimensions in a new DataFrame,  using the Principal Component Analysis (PCA) algorithm.





<img src="https://github.com/halmasieh/Cryptocurrences/blob/main/Resources/PCA.PNG" width="300" height="400"/>






3- Create an elbow curve to find the best value for K from  DataFrame created in step 2.





<img src="https://github.com/halmasieh/Cryptocurrences/blob/main/Resources/Elbow.PNG" />





4- Run the K-means algorithm to predict the K clusters for the cryptocurrencies data, using K-means algorithm.

5- Visualize the distinct groups that correspond to the three principal components you created in step 2, using scatter plot with Plotly Express and hvplot.





<img src="https://github.com/halmasieh/Cryptocurrences/blob/main/Resources/Scatter_Plot.PNG" />





6- Create a table with all the currently tradable cryptocurrencies using the hvplot.table() function.





<img src="https://github.com/halmasieh/Cryptocurrences/blob/main/Resources/Table.PNG" />






7-Write the README report to describe the purpose of what was accomplished.
