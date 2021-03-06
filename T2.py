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
               session=None, mode='concat'):
  
    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.mlp_dim = mlp_dim

    self.mode = mode

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

    if self.mode == 'concat':
        emb_dim += self.emb_dim

    with tf.variable_scope("MLP") as scope:
      self.mlp_W_ = tf.get_variable(
        name="W_", initializer=glorot_uniform(),
        shape=[emb_dim, self.mlp_dim])

      self.mlp_b_ = tf.get_variable(
        name="b_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])

      self.mlp_W = tf.get_variable(
        name="W", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b = tf.get_variable(
        name="b", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      self.x_W = tf.get_variable(
        name="W_x", initializer=glorot_uniform(),
        shape=[self.emb_dim, self.emb_dim])

      self.x_b = tf.get_variable(
        name="b_x", initializer=tf.zeros_initializer(),
        shape=[self.emb_dim])

      self.y_W = tf.get_variable(
        name="W_y", initializer=glorot_uniform(),
        shape=[self.emb_dim, self.emb_dim])

      self.y_b = tf.get_variable(
        name="b_y", initializer=tf.zeros_initializer(),
        shape=[self.emb_dim])

      self.mlp_W_s = tf.get_variable(
        name="W_s", initializer=glorot_uniform(),
        shape=[emb_dim, 1])

      self.mlp_b_s = tf.get_variable(
        name="b_s", initializer=tf.zeros_initializer(),
        shape=[1])

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

    # Now we start defining our graph.

    # 2. Now we define the generative model P(Y | X=x)
    
    # first we need to know some sizes from the current input data
    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]  # longest M
    longest_y = tf.shape(self.y)[1]  # longest N


    # Tile x_embedded to become [B, N*M, emb_dim] by doing 
    # e_j -> ej,...,e_j N times for each j
    xp_shaped1 = tf.reshape(self.x, [batch_size*longest_x, 1])              # Shape: [B*N, 1]
    xp_shaped2 = tf.tile(xp_shaped1, [1, longest_y])                        # Shape: [B*M, N]
    xp_shaped = tf.reshape(xp_shaped2, [batch_size*longest_x*longest_y, 1]) # Shape: [B*M*N, 1]
    self.xp_shaped1 = xp_shaped1
    self.xp_shaped2 = xp_shaped2
    self.xp_shaped = xp_shaped
    # Now embed the properly shaped xp
    self.xp_embedded = tf.nn.embedding_lookup(x_embeddings, xp_shaped)      # Shape: [B*M*N, 1, emb_dim]
    self.xp_embedded = tf.squeeze(self.xp_embedded) # removes the trailing 1-dimension: shape [B*N*M, emb_dim]

    # Tile y_embedded to become [B, N*M, emb_dim] by doing 
    # f_1,...f_N -> f_1,...,f_N, ,...., f_1,...,f_N, total M times
    twos = 2*tf.ones([tf.shape(self.y)[0], 1], tf.int64) # prepend all the sentences with '2', the code for <S>
    yp = tf.concat([twos, self.y[:,:-1]], axis=1, name='concat-twos')
    self.y_p = yp
    yp_shaped1 = tf.tile(yp, [1, longest_x]) # Shape: [B*N, M]
    yp_shaped = tf.reshape(yp_shaped1, [batch_size*longest_x*longest_y, 1]) # Shape: [B*M*N, 1]
    self.yp_shaped1 = yp_shaped1
    self.yp_shaped = yp_shaped
    # Now embed the properly shaped yp
    self.yp_embedded = tf.nn.embedding_lookup(y_embeddings, yp_shaped)      # Shape: [B*M*N, 1, emb_dim]
    self.yp_embedded = tf.squeeze(self.yp_embedded) # removes the trailing 1-dimension: shape [B*N*M, emb_dim]


    self.concatted = tf.concat([ self.xp_shaped, self.yp_shaped ], axis=1)

    # It's also useful to have masks that indicate what
    # values of our batch we should ignore.
    # Masks have the same shape as our inputs, and contain
    # 1.0 where there is a value, and 0.0 where there is padding.
    x_mask  = tf.cast(tf.sign(self.x), tf.float32)    # Shape: [B, M]

    x_mask = tf.expand_dims(x_mask, 2)               # Shape: [B, M, 1]
    x_mask = tf.tile(x_mask, [1, 1, longest_y])      # Shape: [B, M, N]
    x_mask = tf.reshape(x_mask, [batch_size, longest_x*longest_y], name='culprit') # Shape: [B, M*N]

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

    # 2.b P(Y | X, A) = P(Y | X_A)

    # First we make the input to the MLP 2-D.
    # Every output row will be of size Vy, and after a softmax
    # will sum to 1.0.

    # TIM: this is where we either concatenate the two word embeddings
    #      (x and yp) or use a non-linear transformation (gate). We also need
    #      to double the input embedding size if the mode is concat.
    emb_dim = self.emb_dim

    if self.mode == 'concat':
        # Concatenate the word embeddings.
        # Shapes [B, M*N, emb_dim] and [B, M*N, emb_dim] give [B, M*N, 2*emb_dim]
        embedded = tf.concat([ self.xp_embedded, self.yp_embedded ], axis = 1, name='concafly') # Shape [B*N*M, 2*emb_dim]
        emb_dim += self.emb_dim
        
    elif self.mode == 'gate':
        # As a function of the embedding of the previous f, compute a gate value s in [0,1]
        s = tf.matmul(self.yp_embedded, self.mlp_W_s, name='3')
        s = s + self.mlp_b_s
        s = tf.sigmoid(s)

        # Affine transformation followed by tanh
        self.xp_embedded = tf.matmul(self.xp_embedded, self.x_W, name='xp') + self.x_b # Shape [B*N*M, emb_dim]
        self.xp_embedded  = tf.tanh(self.xp_embedded)

        self.yp_embedded = tf.matmul(self.yp_embedded, self.y_W, name='yp') + self.y_b # Shape [B*N*M, emb_dim]
        self.yp_embedded = tf.tanh(self.yp_embedded)
        # Shape [B*N*M, emb_dim]
        embedded = tf.multiply(self.yp_embedded, s, name='scalar1') + tf.multiply(self.xp_embedded, 1 - s)

    self.embedded = embedded
    mlp_input = tf.reshape(embedded, [batch_size * longest_x*longest_y, emb_dim])

    # Here we apply the MLP to our input.
    h = tf.matmul(mlp_input, self.mlp_W_, name='1') + self.mlp_b_  # affine transformation
    h = tf.tanh(h)                                       # non-linearity
    h = tf.matmul(h, self.mlp_W, name='2') + self.mlp_b            # affine transformation [B * M, Vy]

    # Now we perform a softmax which operates on a per-row basis.
    py_xa = tf.nn.softmax(h)
    # py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size])
    py_xa = tf.reshape(py_xa, [batch_size, longest_x*longest_y, self.y_vocabulary_size])

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
    self.py_xa = py_xa
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
    py_xa, acc_correct, acc_total = self.session.run(
      [self.py_xa, self.accuracy_correct, self.accuracy_total],
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
        fprevs = [fprev + (k * longest_y) for k in range(longest_x)]
        probs = py_xa[b, fprevs, y[b,j]] # y[b,j] means only the word f_j in the sentence b
        a_j = probs.argmax()
        p_j = probs[a_j]

        alignments[b, j] = a_j
        probabilities[b, j] = p_j


    return alignments, probabilities, acc_correct, acc_total
