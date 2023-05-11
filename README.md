# This is the code for paper:  Sequential Ensemble Learning for Next Item Recommendation


## Requirement:
python 3.6

tensorflow-gpu==1.14.0

## Run code: 
### STEP 1:
Download the data sets in pan.baidu.com, and unzip it into the path '/datasets/' 
url: https://pan.baidu.com/s/1E5q9zbVYdaXFD_CoYHeOBg?pwd=1234 
code：1234 
### STEP 2:
Generate the base models by run 'basemodel/main.py' for each data set.
Otherwise, you can download the file "Kindle" in pan.baidu.com, and unzip it into the path '/datasets/basemodel/' 
url: https://pan.baidu.com/s/1BxW3bToTWNNStVtuOXinhw?pwd=1234 
code：1234 
### STEP 3:
Run 'main.py' for the proposed method.
## NOTE:

The main.py describes the hyper-parameter of the proposed method and the ablations, which can be summaried as follows:


| method_name        | tradeoff   |  user_module  | model_module| div_module|
| ---------- | :-----------:  | :-----------: |  :-----------: |  :-----------: |
|SEM:        |tradeoff[data]|'SAtt'     |'dynamic'   |'cov'     |
|w/o uDC:    |tradeoff[data]|'static'   |'dynamic'   |'cov'     |
|w/o bDE:    |tradeoff[data]|'SAtt'     |'static'    |'cov'     |
|w/o Div:    |0.0           |'SAtt'     |'dynamic'   |'cov'     |
|w/o TPDiv:  |tradeoff[data]|'SAtt'     |'dynamic'   |'AEM-cov' |
