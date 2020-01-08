r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 128
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "SCENE"
    temperature = .001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences instead of training on the entire text, because backpropagating 
through the entire text will take too much time. 
We do one forward pass through our entire model for each character in the text, which mean passing through all
the layers in the model until reaching the output layer.
And for a sequence of characters, we will do forward pass for each character and pass the hidden state from each 
forward pass to the next character forward pass.
Which means that when we would like to backward through our model to train it, we will need to backpropagte through 
time as well as through layers, and because we forward the hidden state from each character to the next one, we
would backpropagte from the end of our sequence to the beginning of it. Which means (in case we dont split 
the corpus) that we would have to backpropagte through millions of characters which is not doable.

"""

part1_q2 = r"""
Having memory longer than the sequence length is the result of passing the hidden state from each batch to the
next batch in one epoch.
As we can see in the Sequence Batch Sampler function, we split our dataset into batches which complete each other.
Which means that every sample in a batch is the continuation of the same sample from previous batch.
While training, we save the output hidden layers for each batch and passing it to the next batch, which means 
each batch training start with the last batch hidden states (all the previous characters in the text are encoded 
inside the hidden state, which leads to longer memory)

"""

part1_q3 = r"""
Shuffling the order of the batches while training will destroy the hidden state context.
As we stated above, each batch will receive the output hidden states from the previous batch, and by doing so
we can make our model learn longer text than the sequence we give it each time.
If we shuffle the order of batches, we will make our model learn wrong data. Each batch will receive the previous
batch hidden state, but there is no relation between those batches so the hidden state of the previous state
is meaningless to the current batch.
"""

part1_q4 = r"""
1. When training we want our model to output the real result of the predicted character, and we dont want
to give higher values character more "power" by using temperature, because then our model will learn to predict 
characters only when using the same temperature and will calculate the loss wrongly, which means even after training 
there might be multiple characters with high score values, and the model will have low certainty of the predicted char.
While sampling we lower the temperature in order to distinguish between close values characters. This will 
make our model output logical words and sentences (most of the time).

2. High temperature will cause our model scoring to be negligible.
looking at the hot softmax function 

$hot softmax = \frac{e^{\frac{y}{T}}}{\sum_{i}^{ } e^{\frac{y_i}{T}}}$
we can see in the exponent the denominator is T which means that the fraction will approach zero as T gets bigger,
which will make our distribution of characters probabilities to become more and more uniform.

3. Very low temperature will cause a big distinction between 2 very closed value characters, which means 
even if we would like our distribution to be as close to delta function as possible , we would neglect characters which
have very close probabilities as the higher probability char. This happens because the small denominator T , will 
cause the exponent to explode, which will highlight only the highest probability character.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


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


