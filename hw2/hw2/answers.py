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
1. In our case depth of 2 or 4 layers produced the best result. 
Theoretically, increasing the depth should increase our accuracy, because deeper networks have larger number of parameters,
which help the network to achieve better representation of the world.
But there are a few problems that arises when we make our network deeper, one of them is vanishing gradient or gradient explosion,
which causes the network to be untrainable.

2. For the values of 8 or 16 the network was untrainable. The reason those networks were untrainable was because
of vanishing gradients.
Our network consists of convolution layers, and the deeper the network is, the higher the number of convolution layers
we would have to backpropagate through. Each of those layers will cause our gradients to become smaller, resulting
in zero gradient returning to the parameters, which causes the network to learn nothing.
One solution to this problem is skip connection, which allows the gradient to flow freely through the layers,
this is implemented in residual blocks.
Second solution is using better initialization  to our parameters.
For example using Xavier initialization. This will make training possible even in those deep networks.

"""

part3_q2 = r"""
First of all we can see from the results that the networks with depth of 8 didn`t learn anything, this happened 
for the same reasons we stated in the previous question.
We can also see in the graphs that the deeper the network is, the higher the number of filters in each layer it prefers.
We can see that the best result on the test set in the 2-layer network was with 32 filters in each layer,
and the best result on the test set in the 4-layer network was with 256 filters in each layer 
Having more filters will mean having more parameters in our network, which like depth of the network, helps us 
learn more complex function and representation of our problem, but in our network, specially in the 2-layer network,
the complex of the function we can represent is lower because of the low depth, so having more parameters might not
help (some filters might learn the same features and therefor are useless)
"""

part3_q3 = r"""
As we can see from the results, for depth of 9 or 12 layers, the network was untrainable, again for the same reasons 
we stated before (vanishing gradients)
We can see in the results for L=1,2 that for L=2 our network was overfitting our train set, while the network with
L=1 did not. (for L=2 we got 90% accuracy for train set while only around 55% accuracy for test set).
This might happen when we have to many parameters and not deep enough network (the function it can represent is not
complex enough)

"""

part3_q4 = r"""
In contrary to the previous questions , in this part all of our networks were able to train, even the 32-layers network.
This is different than before because of one main reason, which is the use of residual network, which have skip connection
between every few convolution layers that helped the gradients to flow freely from layer to layer.
For this reason the network was able to train.

"""

part3_q5 = r"""
1. We implemented the inception block in our network.
Each block consists of four different paths, the first 2 path are normal convolutions layers with different 
kernel sizes , third path is a pooling path and the last path is the shortcut path.
For each layer we have also added batch normalization layer and dropout layer after each convolution layer.
2. First of all we can see that our architecture has achieved higher accuracy than the networks we previously 
tested. We can also see that due to the skip connection we added in each layer, the network was able to train
even with 36 layers.
"""
# ==============
