import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data

# for TF 1.1
import tensorflow
try:
  from tensorflow.contrib.keras.initializers import glorot_uniform
except:  # for TF 1.0
  from tensorflow.contrib.layers import xavier_initializer as glorot_uniform


class NeuralIBM1Model:
  """Our Neural IBM1 model."""

  def __init__(self, batch_size=8,
               x_vocabulary=None, y_vocabulary=None,
               emb_dim=32, mlp_dim=64,
               session=None):
  
    self.batch_size = batch_size
    self.emb_dim = emb_dim

    self.mlp_dim = mlp_dim

    self.x_vocabulary = x_vocabulary
    self.y_vocabulary = y_vocabulary
    self.x_vocabulary_size = len(x_vocabulary)
    self.y_vocabulary_size = len(y_vocabulary)

    self._create_placeholders()
    self._create_weights()
    self._build_model()

    self.saver = tf.train.Saver()
    self.session = session

  def _create_placeholders(self):
    """We define placeholders to feed the data to TensorFlow."""
    # "None" means the batches may have a variable maximum length.
    self.x  = tf.placeholder(tf.int64, shape=[None, None],
                             name = "english")
    self.yp = tf.placeholder(tf.int64, shape=[None, None],
                             name = "prev_french")
    self.y  = tf.placeholder(tf.int64, shape=[None, None],
                             name = "french")

  def _create_weights(self):
    """Create weights for the model."""
    # TIM: we need to double the input embedding size if the mode is concat.
    emb_dim = self.emb_dim

    with tf.variable_scope("MLP") as scope:
      # first layer for t
      self.mlp_Wt_ = tf.get_variable(
        name="Wt_", initializer=glorot_uniform(),
        shape=[emb_dim, self.mlp_dim])

      self.mlp_bt_ = tf.get_variable(
        name="bt_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])

      # first layer for i and s
      self.mlp_Wis_ = tf.get_variable(
        name="Wis_", initializer=glorot_uniform(),
        shape=[emb_dim, self.mlp_dim])

      self.mlp_bis_ = tf.get_variable(
        name="bis_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])

      # layer for translation P(F|E)
      self.mlp_W_t = tf.get_variable(
        name="W_t", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b_t = tf.get_variable(
        name="b_t", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      # layer for insertion P(F|Fprev)
      self.mlp_W_i = tf.get_variable(
        name="W_i", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b_i = tf.get_variable(
        name="b_i", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      # layer for laten gate S, P(S|Fprev)
      self.a_W = tf.get_variable(
        name="a_W", initializer=tf.random_normal_initializer(),
        shape=[mlp_dim, self.z_dim])

      self.a_b = tf.get_variable(
        name="a_b", initializer=tf.random_normal_initializer(),
        shape=[self.z_dim])

      self.b_W = tf.get_variable(
        name="b_W", initializer=tf.random_normal_initializer(),
        shape=[mlp_dim, self.z_dim])

      self.b_b = tf.get_variable(
        name="b_b", initializer=tf.random_normal_initializer(),
        shape=[self.z_dim])

      self.alpha_W = tf.get_variable(
        name="alpha_W", initializer=tf.random_normal_initializer(),
        shape=[mlp_dim, self.z_dim])

      self.alpha_b = tf.get_variable(
        name="alpha_b", initializer=tf.random_normal_initializer(),
        shape=[self.z_dim])

      self.beta_W = tf.get_variable(
        name="beta_W", initializer=tf.random_normal_initializer(),
        shape=[mlp_dim, self.z_dim])

      self.beta_b = tf.get_variable(
        name="beta_b", initializer=tf.random_normal_initializer(),
        shape=[self.z_dim])


  def save(self, session, path="model.ckpt"):
    """Saves the model."""
    return self.saver.save(session, path)

  def _build_model(self):
    """Builds the computational graph for our model."""

    # 1. Let's create a (source) word embeddings matrix.
    # These are trainable parameters, so we use tf.get_variable.
    # Shape: [Vx, emb_dim] where Vx is the source vocabulary size
    x_embeddings = tf.get_variable(
      name="x_embeddings", initializer=tf.random_uniform_initializer(),
      shape=[self.x_vocabulary_size, self.emb_dim])
    y_embeddings = tf.get_variable(
      name="y_embeddings", initializer=tf.random_uniform_initializer(),
      shape=[self.y_vocabulary_size, self.emb_dim])

    emb_dim = self.emb_dim

    # Now we start defining our graph.

    # ###############################################
    # This is the **inference** network q_\phi(Z | x)
    #
    #  it predicts for each x a d-dimensional vector of means and a vector of (log) variances
    #  it does so from x's 1-hot encoding
    #  thus the first step is to embed x
    
    # first we need to know some sizes from the current input data
    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]  # longest M
    longest_y = tf.shape(self.y)[1]  # longest N

    # Input yp
    twos = 2*tf.ones([tf.shape(self.y)[0], 1], tf.int64) # prepend all the sentences with '2', the code for <S>
    yp = tf.concat([twos, self.y[:,:-1]], axis=1, name='concat-twos')
    self.y_p = yp
    yp_embedded = tf.nn.embedding_lookup(y_embeddings, yp) # Shape: [B, N, emb_dim]
    self.yp_embedded = yp_embedded

    # Input x
    x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x) # Shape: [B, M, emb_dim]
    

    # It's also useful to have masks that indicate what
    # values of our batch we should ignore.
    # Masks have the same shape as our inputs, and contain
    # 1.0 where there is a value, and 0.0 where there is padding.
    x_mask  = tf.cast(tf.sign(self.x), tf.float32)    # Shape: [B, M]
    y_mask  = tf.cast(tf.sign(self.y), tf.float32)    # Shape: [B, N]
    x_len   = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
    y_len   = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]

    # 2.a Build an alignment model P(A | X, M, N)

    # This just gives you 1/length_x (already including NULL) per sample.
    # i.e. the lengths are the same for each word y_1 .. y_N.
    lengths  = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
    pa_x     = tf.div(x_mask, tf.cast(lengths, tf.float32))   # Shape: [B, M]

    # We now have a matrix with 1/M values.
    # For a batch of 2 setencnes, with lengths 2 and 3:
    #
    #  pa_x = [[1/2 1/2   0]
    #          [1/3 1/3 1/3]]
    #
    # But later we will need it N times. So we repeat (=tile) this
    # matrix N times, and for that we create a new dimension
    # in between the current ones (dimension 1).
    pa_x  = tf.expand_dims(pa_x, 1)  # Shape: [B, 1, M]

    #  pa_x = [[[1/2 1/2   0]]
    #          [[1/3 1/3 1/3]]]
    # Note the extra brackets.

    # Now we perform the tiling:
    pa_x  = tf.tile(pa_x, [1, longest_y, 1])  # [B, N, M]

    # pa_x = tf.tile(y_embedded, [1, 1, longest_y])

    # Result:
    #  pa_x = [[[1/2 1/2   0]
    #           [1/2 1/2   0]]
    #           [[1/3 1/3 1/3]
    #           [1/3 1/3 1/3]]]

    # The MLP

    # This is for P(F|E), translation t
    mlp_input = tf.reshape(x_embedded, [batch_size * longest_x, emb_dim])
    h_t = tf.matmul(mlp_input, self.mlp_Wt_, name='x1') + self.mlp_bt_ # Shape [B*M, emb_dim]
    h_t  = tf.tanh(h_t)
    h_t = tf.matmul(h_t, self.mlp_W_t, name='x2') + self.mlp_b_t # Shape [B*M, emb_dim]
    # Now we perform a softmax which operates on a per-row basis.
    py_xa = tf.nn.softmax(h_t)
    # This is P(F|E)
    py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size]) # Shape [B, M, Vy]

    # This is for P(F|Fprev) and P(C|Fprev), insertion i and collocation c
    # Note: Shared first layer!
    mlp_input = tf.reshape(yp_embedded, [batch_size * longest_y, emb_dim])
    h_is = tf.matmul(mlp_input, self.mlp_Wis_, name='y1') + self.mlp_bis_ # Shape [B*N, emb_dim]
    h_is  = tf.tanh(h_is)
    # This is P(F|Fprev) insertion i
    h_i = tf.matmul(h_is, self.mlp_W_i, name='y2') + self.mlp_b_i # Shape [B*N, emb_dim]
    py_y  = tf.nn.softmax(h_i) # Shape: [B*N, Vy]
    py_y = tf.reshape(py_y, [batch_size, longest_y, self.y_vocabulary_size]) # Shape [B, N, Vy]
    # This is s(Fprev) for P(C|Fprev) = Bern(s(Fprev))
    h_s = tf.matmul(h_is, self.mlp_W_s, name='3')
    h_s = h_s + self.mlp_b_s
    s = tf.sigmoid(h_s)
    s = tf.squeeze(s) # get rid of trainling 1-dimension
    s = tf.reshape(s, [batch_size, longest_y])

    alpha = tf.matmul(h_is, self.alpha_W) + self.alpha_b  # [B * M, z_dim]
    beta = tf.matmul(h_is, self.beta_W) + self.beta_b  # [B * M, z_dim]
    # IMPORTANT: used for sampling gate values s:
    a = tf.matmul(h_is, self.a_W) + self.a_b  # [B * M, z_dim]
    b = tf.matmul(h_is, self.b_W) + self.b_b  # [B * M, z_dim]


    # Our sampled S is a **deterministic** function of the random noise (u)
    # this pushes all sources of non-determinism out of the computational graph
    # which is very convenient
    # In formula: s = (1-u^{1/alpha})^{1/beta}
    u = tf.random_uniform(tf.shape(z_b), minval=0, maxval=1, dtype=tf.float32)  # [B * M, h_dim]
    s = tf.pow(tf.add(-tf.pow(u, tf.reciprocal(alpha)), 1), tf.reciprocal(beta)) # hopefully this works


    # ##############################################
    # This is the *generative* network
    #  it conditions on our sampled z to predict the parameters of a Categorical over the vocabulary

    # Here we employ one non-linear layer (but this is optional)
    ### CHANGE ###
    # h_dec = tf.matmul(s, self.y_W) + self.y_b  # [B * M, h_dim]
    # OR: (confusion!)
    # h_dec = tf.matmul(s_ELBO, self.y_W) + self.y_b  # [B * M, h_dim]
    ###
    h_dec = tf.matmul(z, self.y_W) + self.y_b  # [B * M, h_dim]
    h_dec = tf.tanh(h_dec)  # Shape: [B * M, h_dim]
    # and these are our logits (the input to a softmax)
    # tensorflow prefers to use logits to compute the cross entropy loss
    # but see that this is just a code optimisation probably motivated solely by numerical stability
    logits = tf.matmul(h_dec, self.softmax_W) + self.softmax_b  # Shape: [B * M, Vx]

    # ###############################################
    # This is the MC estimate of the (negative) ELBO (because we do minimisation here)
    #  it includes the MC estimate of the negative log likelihood

    # Sample gate value s. note weird tf construction for beta disrtb.
    # see https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/Beta
    # Beta = tf.contrib.distributions.Beta(a,b) # constructs distributions same shape as a (and b)
    # s = Beta.sample() # generates a single sample for each of the distributions

    euler = 0.5772156649
    approx = tf.add_n([tf.reciprocal(m + alpha*beta) * Beta(m * tf.reciprocal(alpha), beta)) for m in range(1,4)])
    first = tf.multiply(tf.div(alpha - a, alpha), -euler - tf.digamma(beta) - tf.reciprocal(beta))
    second = tf.log(tf.multiply(alpha, beta)) + tf.log(1) - tf.multiply(beta - 1, tf.reciprocal(beta)) # Beta(a,b) missing! Cannot find it in TF...
    third = tf.multiply(tf.multiply(b - 1, beta), approx)
    kl = first + second + third

    kl = tf.reshape(kl, tf.shape(self.x))  # reshape back to [B, M]
    # we sum KL of actual words in a sentence
    #  (that's why we multiply timesteps by a mask)
    #  and take mean over samples
    self.kl = tf.reduce_mean(tf.reduce_sum(kl * x_mask, axis=1), axis=0)

    # The total loss is the negative MC estimate of the ELBO
    self.loss = self.ce + self.kl

    # #########################################################
    # Prediction
    #
    # Note that while training is stochastic (we sample z by sampling epsilon
    #  and computing Z= mu(x) + epsilon * sigma(x)
    #  for predicted mu(x) and sigma(x)
    #
    # we will simplify *predictions* by making them deterministic
    #  that is, for *predictions only* we will pretend z can be represented by
    #  the predicted mean, i.e. Z=mu(x)
    #
    #  Why is this a simplification?
    #  * once we get an assignment to Z, the generative network applies
    #    a nonlinear layers before predicting the final softmax
    #    this means that
    #      E[softmax(f_\theta(z))\ != softmax(f_\theta(E[z]))
    #    where E[z] = mu(x)
    #
    #  The principled thing to do is to sample a few assignments (e.g. 100)
    #   and run the generative model on each assignment
    #   then use a probabilistic disambiguation rule (e.g. most-probable-sample,
    #   or MBR, etc.).
    #  Here instead we simply take the mean as a hopefully good approximation.


    # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)

    # Here comes a rather fancy matrix multiplication.
    # Note that tf.matmul is defined to do a matrix multiplication
    # [N, M] @ [M, Vy] for each item in the first dimension B.
    # So in the final result we have B matrices [N, Vy], i.e. [B, N, Vy].
    #
    # We matrix-multiply:
    #   pa_x       Shape: [B, N, *M*]
    #       pa_x       Shape: [B, N, *N*M*]
    # and
    #   py_xa      Shape: [B, *M*, Vy]
    #       py_xa      Shape: [B, *N*M*, Vy]
    # to get
    #   py_x  Shape: [B, N, Vy]
    #
    # Note: P(y|x) = prod_j p(y_j|x) = prod_j sum_aj p(aj|m)p(y_j|x_aj)
    py_x = tf.matmul(pa_x, py_xa, name='3')  # Shape: [B, N, Vy]

    # Read the equation in Theory 2.2 carefully. Then you will see that this is correct.
    s_tiled = tf.expand_dims(s, 2) # Shape: [B, N, 1]
    s_tiled = tf.tile(s_tiled, [1, 1, self.y_vocabulary_size]) # Shape: [B, N, Vy]
    # Here we marginalise over S
    py_x = tf.multiply(s_tiled, py_x, name='s1') + tf.multiply(1-s_tiled, py_y, name='s2')



    # This calculates the accuracy, i.e. how many predictions we got right.
    predictions = tf.argmax(py_x, axis=2)
    acc = tf.equal(predictions, self.y)
    acc = tf.cast(acc, tf.float32) * y_mask
    acc_correct = tf.reduce_sum(acc)
    acc_total = tf.reduce_sum(y_mask)
    acc = acc_correct / acc_total

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(self.y, [-1]),
      logits=tf.log(tf.reshape(py_x,[batch_size * longest_y, self.y_vocabulary_size], name='lastfucker')),
      name="logits"
    )
    cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
    cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
    cross_entropy = tf.reduce_mean(cross_entropy, axis=0)


    self.pa_x = pa_x
    self.py_x = py_x
    self.py_y = py_y
    self.py_xa = py_xa
    self.s = s
    self.loss = cross_entropy
    self.predictions = predictions
    self.accuracy = acc
    self.accuracy_correct = tf.cast(acc_correct, tf.int64)
    self.accuracy_total = tf.cast(acc_total, tf.int64)


  def evaluate(self, data, ref_alignments, batch_size=4):
    """Evaluate the model on a data set."""

    ref_align = read_naacl_alignments(ref_alignments)

    ref_iterator = iter(ref_align)
    metric = AERSufficientStatistics()
    accuracy_correct = 0
    accuracy_total = 0

    for batch_id, batch in enumerate(iterate_minibatches(data, batch_size=batch_size)):
      x, y = prepare_data(batch, self.x_vocabulary, self.y_vocabulary)
      y_len = np.sum(np.sign(y), axis=1, dtype="int64")

      align, prob, acc_correct, acc_total = self.get_viterbi(x, y)
      accuracy_correct += acc_correct
      accuracy_total += acc_total

      for alignment, N, (sure, probable) in zip(align, y_len, ref_iterator):
        # the evaluation ignores NULL links, so we discard them
        # j is 1-based in the naacl format
        pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
        metric.update(sure=sure, probable=probable, predicted=pred)

    accuracy = accuracy_correct / float(accuracy_total)
    return metric.aer(), accuracy


  def get_viterbi(self, x, y):
    """Returns the Viterbi alignment for (x, y)"""

    feed_dict = {
      self.x: x, # English
      self.y: y, # French
    }

    # run model on this input
    py_xa, py_y, s, acc_correct, acc_total = self.session.run(
      [self.py_xa,
       self.py_y,
       self.s,
       self.accuracy_correct, 
       self.accuracy_total],
       feed_dict=feed_dict)

    # things to return
    batch_size, longest_y = y.shape
    _, longest_x = x.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")

    for b, sentence in enumerate(y):
      for j, french_word in enumerate(sentence):
        if french_word == 0:  # Padding
          break
        fprev = j
        sj = s[b,j]
        # if b in range(20): print(sj)
        c = int(np.random.uniform() < sj) # sample c ~ Bernouilli(sj)
        if c == 0: # then we align
            probs = py_xa[b, : , y[b,j]] # y[b,j] means only the word f_j in the sentence b
            a_j = probs.argmax()
            p_j = probs[a_j]
        if c == 1: # then we `insert` (i.e. NULL align - see NLP2 blog post)
            # if b in range(20): print('Null aligned')
            a_j = 0 # NULL align
            p_j = 1 # not important

        alignments[b, j] = a_j
        probabilities[b, j] = p_j


    return alignments, probabilities, acc_correct, acc_total
