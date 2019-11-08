r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing the k value can lead to improved generalization for unseen data, because it allows the model to base its
prediction on more training samples. For example, in the case of digit recognition, we can compare the test samples
with more trained digits, and so if we assume that the given sample is more similar to the correct part of the training
data than other parts, then a bigger k will associate the test sample with more training samples of the correct category.

However, if the k value is too big, then the k nearest neighbors can (and probably will) contain training samples of the 
wrong categories, and therefore can lead to incorrect predictions. As an extreme example, if every digit has n training 
samples, and we choose k=3n , then for every prediction we will have at least 2n nearest neighbors of wrong categories, 
and that can reduce the accuracy of the model by a lot.

"""

part2_q2 = r"""
**Your answer:**
1. Testing every model on the data which it was trained on is a bad idea, because it ignores the need for generalization.
In the case of kNN, testing on the training data will always result in 100% accuracy for k=1, because the only nearest
neighbor of every sample will be itself, and therefore it will already "know" the correct answer.
2. If we were to pick the model based on its accuracy on the test-set after it was trained on train-set, we might
choose a model that was "lucky", in the sense that its unique k value is better than the others at predicting the test-set
based on the train-set, when it is actually worse than other at predicting other samples.
However, when splitting the train-set to k parts and then training each model some parts of the train-set, and testing it 
on the other parts, it reduces the amount of "luck" that a model can have, because it tests it in more scenarios, and 
therefore we pick a model which has a better ability to generalize.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The selection of delta is arbitrary because the magnitude of our classes scores has no meaning in svm loss function,
which only refers to the distance of the predicted class score to the real class score.
This means that as long as the predicted class score is at least delta smaller than the real class score the loss 
is zero.
Using regularization on our weights will makes our weights smaller, and therefor also making the distance between 
the scores smaller, which gives delta more effect over our loss and scale our weights up.
This means the value we give to delta, which is the margin we want between our scores is meaningless, because
the weights values and class scores can become smaller or larger arbitrarily, which mean the exact delta value
has no real meaning except the final size of the weights.

"""

part3_q2 = r"""
**Your answer:**
1. By visualizing the linear model weights for each class, we can see which input features will result in an higher 
activation score for each class.
Meaning we can look at the image pixels, and determine which pixels or pattern of the image have more effect by
the brightness (higher pixel value) of each pixel.
We should see in each class`s weights image which input the current class is looking for in the input image.
The classification errors we got using our model occurs because the input image had similar features to the
predicted class, by looking at the weights image, in some cases we can interpret the bright parts as more than 1 number. 
2. This interpretation is different from the KNN model, in the KNN model our training process includes only memorizing
the dataset and the model does not learn any unknown data, while in the linear model we try to learn which 
features belongs to which class and makes our classification according to it.
"""

part3_q3 = r"""
**Your answer:**
1. I would say the learning rate is Good.
At our first attempt we tried learning rate which equals to 0.1, and we found out that our loss in a bit too "jumpy"
(it started by decreasing and every few epochs the loss would drastically become larger for one epoch).
we realised that our learning rate was too high which caused our model to "jump" over the minimum point.
After decreasing the learning rate by a factor of 10 the loss became more monotonous stable.
If we used a smaller learning rate we would make smaller steps at the direction of the minimum, and our loss values 
would decrease slower and it would take more epochs to reach the same loss value we got in our setting.
2. I would say that the model is slightly overfitted to the training set.
We can see in the graph that the training accuracy is higher than the validation accuracy for almost the entire training
process, and the improvement of the training accuracy is slightly better than the validation accuracy,
which means our model is overfitting to our training dataset.
But we can also see that the margin between the training accuracy and validation accuracy is not very big, which
means that the overfitting is not as significant.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
