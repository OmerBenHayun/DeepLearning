r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp['h1'] = 32
    hp['h2'] = 32
    hp['batch_size'] = 64
    hp['gamma'] = 0.999
    hp['learn_rate'] = 8e-3
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp['h1'] = 64
    hp['h2'] = 32
    hp['batch_size'] = 64
    hp['gamma'] = 0.999
    hp['learn_rate'] = 8e-3
    hp['delta'] = 0.7
    hp['beta'] = 0.7
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

Intuitively, subtracting a decent baseline from the reward in the calculation of the gradients, leads to a smaller rewards 
(in absolute value), which leads to smaller gradients overall, which means reduced variance and more stable training.

We can look at an example batch, with some of the trajectories leading to a reward of 1, and the rest of the trajectories 
leading to a reward of 100. Clearly, we will want our algorithm to learn only the trajectories that lead to the 100 reward,
but since both rewards are positive, then the network will adjust its weights towards both trajectories. However, if there 
are many more samples of reward 1, then the network will most likely fit to take trajectories that lead to this reward, which
is not what we want.
In order solve this problem, we can subtract a baseline, such that the reward 1 will become a negative reward, and the reward 100
will remain a positive reward. This way, the network will actually move away from the trajectories that lead to reward 1, and 
towards the trajectories that lead to reward 100, which is what we are looking for.

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
