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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
