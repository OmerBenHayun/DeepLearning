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
    hypers['h_dim'] = 256
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT"
    temperature = .001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences instead of training on the entire text, because of 2 reasons :
1. Backpropagating through the entire text will take too much time. 
We do one forward pass through our entire model for each character in the text, which mean passing through all
the layers in the model until reaching the output layer.
And for a sequence of characters, we will do forward pass for each character and pass the hidden state from each 
forward pass to the next character forward pass.
Which means that when we would like to backward through our model to train it, we will need to backpropagte through 
time as well as through layers, and because we forward the hidden state from each character to the next one, we
would backpropagte from the end of our sequence to the beginning of it. Which means (in case we dont split 
the corpus) that we would have to backpropagte through millions of characters which is not doable.
2. To avoid saving the entire corpus in memory.
By splitting the corpus into sequences, we can keep only the current sequences in the memory instead of saving
the entire corpus. 
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
    hypers['batch_size'] = 1
    hypers['h_dim'] = 32
    hypers['z_dim'] = 10
    hypers['x_sigma2'] = 1.0
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

When we calculate the reconstruction loss (= data loss) part of the total loss, we divide the MSE between
the original image and the reconstructed image by ${\sigma}^2$ . Therefore, the hyperparameter ${\sigma}^2$ is
used to adjust the weight we give to the reconstruction loss in the total loss.

If the value of ${\sigma}^2$ is low, then the weight given to the data loss is high, and the consequence
of that is that the output of the network (images reconstructed from latent space vector) will be more
similar to the images that it was trained on.

If the value of ${\sigma}^2$ is high, then the weight given to the data loss is low, and the consequence
of that is that the output images will be more different from the training images, which means a more diverse
space of output images, however if ${\sigma}^2$ is too small, then the output might not show the object that we
wanted to represent.

"""

part2_q2 = r"""
**Your answer:**

1. The VAE loss function is the sum of 2 measures:
    - a reconstruction loss term - the MSE between the input image and the output image
    - a KL-divergence term - a measure of how each component's normal distribution in the latent space is different from $N(0,1)$.

2. The KL term in the loss function penalizes distributions that are different from $N(0,1)$ , and that leads to the distributions
of the latent representations being close to the center of the latent space, and with deviations that are close to 1.

3. The benefit of the latent representation distributions being encouraged to be close to the center of the latent space is that 
when they are grouped, there is not much space between them, which means that the latent space can be considered continuous, and 
that allows us to sample any vector from the distribution $N(0,1)$ , decode it and receive a viable result image.

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
    hypers['batch_size'] = 32
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.3
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['weight_decay'] = 0.02
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['weight_decay'] = 0.02
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)

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
1.When training GAN model to generate images we shouldn`t stop training when the generator loss reach certain threshold.
We need to remember that our generator and discriminator are training together, and against each other.
Which means that the generator loss might be small due to untrained discriminator.
Stopping the training when we reach some threshold, will result in under-trained generator.
2.The discriminator loss consists of two losses, one is due to the generated images from the generator, and the other 
one is due to the real images given to the discriminator.
The generator tries to minimize it`s loss and in result the first loss in the discriminator loss, must get bigger.
Which means that if the discriminator loss remains constant, the second loss (due to the real images) is getting lower,
which means that the discriminator can distinguish real images with more accuracy but can distinguish fake
images with lower accuracy (will classify real images as real with more accuracy, and fake images as fake with less
accuracy).

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


