# OCCER
One Class Classification By Ensembles Of Regression Models


## Usage

    python occer-realvalueddata.py filename.csv



- Program needs only one argument. Either a csv file or an arff file. 
- In the case of csv file, no column names should be present. Last column in the csv file represents the label with values **1 for minority** and **0 for majority**. Same convention for labels is used for arff files.


## What is OCCER?

> OCCER algorithm converts a OCC (One class classification problem) into many regression problems in the original feature space such that each feature of the original feature space is used as the target variable in one of the regression problems. The errors of regression of a data point by all the regression models are used to compute the outlier score of the data point.


## Requirements
- python 3
- pandas
- [pyod](https://github.com/yzhao062/pyod)
- sklearn
- keras
- tensorflow
- liac-arff


## Work with other regression methods ?

       def runColumnWiseRegressorAlgorithm()

Add a new regression algorithm of interest in above method by adding an if clause.

## How to cite ?
yet to come
