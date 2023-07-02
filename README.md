<h1>21001556g Assignment 1, Q2 and Q3</h1>

<h2>Overview</h2>

-- Executable Program:
run main_{question_number}_{model_name}.py

-- Execution Logs:
/Execution Logs

-- Result including plotted graphs and .pth files:
/outputs

-- Models code:
/models

-- Points to note:
1. In Mac with ARM chips like M1, M2, CUDA is not supported, therefore, "MPS" is used instead
2. *Some online posts claim that "MPS with PyTorch" will cause more loss during training which I dont have another Windows device to compare


<h2>Questions Answer</h2>
Q2a. Graphs are plotted in /outputs folder. Resnet18 and GoogleNet shows good performance and convergence for train and test accuracy.
     Vgg11 performs badly in first ~20 epoches but shows the trend of convergence in accuracy.

Q2b. Adam 's learning curve is steeper than nesterov but the outcome is close.
     rmsprop performs poor in the first ~35 epoches and steeply increase in accuracy and loss. The learning process is unsatisfied 

Q2c. Increased the learning rate from 1e-3 to 0.1 and also set a constant learning rate improves the learning curve 

Q3a. Based on the previous checkpoints, the model can provides a ~80% accuracy and more training data, the better performance (~1-2 % increment).

Q3b. Resnet18 and Vgg16, both demonstrated a 1-2% increment in accuracy

Q3c. Idea is to increase the CONV layers and then pass to the FC layers. The product will be Resnet34-like model.


