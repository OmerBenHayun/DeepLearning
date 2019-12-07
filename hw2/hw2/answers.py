r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 3e-3
    reg = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 4e-3
    lr_momentum = 5e-4
    lr_rmsprop = 2e-4
    reg = 1e-4
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 1e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. The graphs of no-dropout vs dropout match our expectations, because we expected that the lowest train_loss and the highest train_acc will be
achieved when using no dropout, since without dropout regularization, the network can easily overfit on the training data. 
In contrast, we also expected that the results on the testing data would be better when training with dropout, and we can see that the test_loss 
graphs of the networks with dropout are lower than the test_loss graph of dropout=0. 
Regarding the test_acc, we can see that indeed when using dropout=0.4 we get higher test_acc than when not using dropout, however the test_acc
of dropout=0.8 is still lower than of dropout=0.

2. In our experiment we tested 2 dropout levels: 0.4 (low) and 0.8 (high) .
We might think that with higher dropout we will get better results on our testing data, since the network will be forced to find better 
representation of the data, however we can see from the graphs that we actually got worse results on the high dropout setting compared to
the low dropout setting in all 4 metrics. It is because we are using a relatively small MLP (400 hidden features), and so when using
dropout=0.8 the network is left with too little expression power to properly learn the classification. We can also see its limited power by
the flatlining of the test_loss graph for dropout=0.8 .
"""

part2_q2 = r"""
It is possible for the test loss to increase while the test accuracy also increases in the theoretical situation where our objective is to
classify objects (for example), and the model's confidence in decisions, where it had high confidence, decreases, which causes the increase 
in the loss value, but not to the point where it classifies them wrongly. Also, at the same time it learns how to correctly classify other 
test samples that were previously classified incorrectly, with low confidence which increases the test accuracy.
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

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
