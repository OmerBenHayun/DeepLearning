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

This is a valid approximation because of the relation between the state-value function and the action-value function.
As we know we can define the state-value to be an average over the possible actions of the action-value of each action:
$V(s) = \sum_a \pi(a|s) * q(s,a)$
state-value is the sum for each action, the probability of taking that action multiplied by the action-value.
(our policy is the probability of taking action a in state s)
This means we can use our q-values (action-values) in order to approximate the state-value, using the relation we wrote.

"""


part1_q3 = r"""

1. 
- **loss_p** graph: The loss_p term is actually the minus of the mean of the losses over every batch, so when we see the graph
increasing, it means that the loss is actually going down. We can see that `vpg` and `epg` start off from a very high loss, and
gradually lowering at a similar rate, which means that they are learning and improving their policy.
The other 2 graphs: `bpg` and `cpg` look very similar to one another, and they both seem to
be almost 0. The difference in each pair is that one uses entropy, and the other doesn't. By the similar graphs, we can understand 
that the addition of an entropy loss term doesn't change the training process very much. In contrast, the different between the 
the models that use a baseline, and those who doesn't is very big. Since in `bpg` and `cpg` we subtract a baseline from the reward,
the loss is smaller in its magnitude compared to the case without the subtraction. It can explain why the loss graphs of the models
that use a baseline seem very close to 0 as compare to the others.

- **baseline** graph: The `bpg` and `cpg` baselines are increasing with time, because as the networks learn, they are able to reach
better states and rewards, and therefore the q-values of the states also increase, and the baselines are only the mean of the
q-values at each batch, so they also increase. It is wanted behavior, because in order to reduce the variance, we want the baseline
to be in a similar magniture to the rewards, so if the rewards increase, we also want the baseline to increase.

- **loss_e** graph: The calculation for the entropy loss is minus the entropy, so the increase in the loss graphs, means that the
entropy actually decreases as the networks are trained. When the entropy is low, it means that the action probability distribution
is far from uniform, which can be interpreted as the network being "sure" of which action it should take. It can also be seen as a
measure of convergence, such that when the entropy is low, the network knows which action it should take. By this interpretation, 
we can see that `epg` converges faster than `cpg`, which shows that the addition of a baseline actually improves the training process.

- **mean_reward** graph: We can see that all 5 graphs show an increase over time, which is a good sign that shows that all 4 models
worked in learning to solve the given problem to some extent.
We can see that the 4 graphs can be divided to 2 batches: the first being `vpg` and `epg`, and the second being `bpg` and `cpg`. 
The 2 models in each batch have achieved very similar results. Since the difference between the 2 models in each batch is only 
the addition of an entropy loss term, we can understand that this addition has very little effect on the model's success. It is 
probably the case, because in the problem we dealt with, the action space is very small, and so there isn't a risk that we will 
ignore some actions. In contrast, the addition of baseline, which differentiates the 2 batches, leads to better mean rewards, 
which proves the theoretical explanations we have given in question 1.


2. 

We can see differences in the `loss_p` graph, the `loss_e` graph and the `mean_reward` graph.
In the `loss_p` graph we can see that the policy loss achieved by the actor-critic method noticeably lower than that achieved by
the regular policy gradients method. It can be explained by the added complexity in the AAC method, which allows for a better 
approximation of the state's values, and therefore it allows the policy network to 'steer' towards better states than `cpg`.
When we look at the `loss_e` graph, we remember that the entropy loss is multiplied by beta, which is 0.5 for `cpg` and 0.7 for
`aac`. We can take a few points, and see that the entropy loss of `cpg` is indeed almost exactly the entropy loss of `aac` when 
these factors are taken into account, and that tells us that the difference between the methods doesn't change the entropy in the 
distributions.
We can see in the `mean_reward` graph that the results of `cpg` and `aac` are very similar. It is probably the case because the 
problem we dealt with is a relatively simple, and so the actor-critic architecture, which provides more power to the training 
process, doesn't show a vast improvement compared to the normal policy gradients with a baseline, which is a simpler method.

"""
