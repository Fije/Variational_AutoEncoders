import numpy as np
import tensorflow as tf
import random
from pprint import pprint
from utils import iterate_minibatches, prepare_data, smart_reader, bitext_reader
import matplotlib.pyplot as plt


class NeuralIBM1Trainer:
  """
  Takes care of training a model with SGD.
  """

  def __init__(self, model, train_e_path, train_f_path,
               dev_e_path, dev_f_path, dev_wa,
               test_e_path, test_f_path, test_wa,
               num_epochs=5,
               batch_size=16, max_length=30, lr=0.1, lr_decay=0.001, session=None,
               max_num=np.inf,
               add_xnull=True):
    """Initialize the trainer with a model."""
    
    self.model = model
    self.train_e_path = train_e_path
    self.train_f_path = train_f_path
    self.dev_e_path = dev_e_path
    self.dev_f_path = dev_f_path
    self.dev_wa = dev_wa
    self.test_e_path = test_e_path
    self.test_f_path = test_f_path
    self.test_wa = test_wa

    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.max_length = max_length
    self.lr = lr
    self.lr_decay = lr_decay
    self.session = session

    print("Training with B={} max_length={} lr={} lr_decay={}".format(
        batch_size, max_length, lr, lr_decay))

    self._build_optimizer()

    # This loads the data into memory so that we can easily shuffle it.
    # If this takes too much memory, shuffle the data on disk
    # and use bitext_reader directly.
    self.corpus = list(bitext_reader(
        smart_reader(train_e_path,max_num=max_num),
        smart_reader(train_f_path,max_num=max_num),
        max_length=max_length))
    # print("Training on {} sentences".format(len(self.corpus)))
    self.dev_corpus = list(bitext_reader(
        smart_reader(dev_e_path),
        smart_reader(dev_f_path)))
    self.test_corpus = list(bitext_reader(
        smart_reader(test_e_path),
        smart_reader(test_f_path)))

  def _build_optimizer(self):
    """Buid the optimizer."""
    self.lr_ph = tf.placeholder(tf.float32)
    # Uncomment this to use simple SGD instead (uses less memory, converges slower)
    #self.optimizer = tf.train.GradientDescentOptimizer(
    #  learning_rate=self.lr_ph).minimize(self.model.loss)

    # use Adam optimizer
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.lr_ph).minimize(self.model.loss)

  def train(self):
    """Trains a model."""

    dev_likelihoods = []
    train_likelihoods = []
    dev_AERs = []
    test_AERs = []
    other_likelihoods = []

    steps = 0

    for epoch_id in range(1, self.num_epochs + 1):

      # shuffle data set every epoch
      print("Shuffling training data")
      # random.shuffle(self.corpus)

      loss = 0.0
      accuracy_correct = 0
      accuracy_total = 0
      epoch_steps = 0

      for batch_id, batch in enumerate(iterate_minibatches(
          self.corpus, batch_size=self.batch_size), 1):

        # Dynamic learning rate, cf. Bottou (2012), Stochastic gradient descent tricks.
        lr_t = self.lr * (1 + self.lr * self.lr_decay * steps)**-1

        x, y = prepare_data(batch, self.model.x_vocabulary,
                            self.model.y_vocabulary,
                            add_xnull=False) # No NULL needed. Insertion takes care of that.

        # input to the TF graph
        feed_dict = {
          self.lr_ph:    lr_t,
          self.model.x:  x,
          self.model.y:  y
        }

        # things we want TF to return to us from the computation
        fetches = {
          "optimizer"   : self.optimizer,
          "loss"        : self.model.loss,
          "acc_correct" : self.model.accuracy_correct,
          "acc_total"   : self.model.accuracy_total,
          "pa_x"        : self.model.pa_x,
          "py_xa"       : self.model.py_xa,
          "py_x"        : self.model.py_x,
          "yp"          : self.model.y_p,
        }

        res = self.session.run(fetches, feed_dict=feed_dict)
        # print(y)
        # print()
        # print(res["yp"])
        # print()
        # pprint(res["yp_shaped1"])
        # print("y-shaped")
        # pprint(res["yp_shaped"])
        # print()
        # pprint(res["yp_embedded"])
        # print()
        # print()
        # print(x)
        # print()
        # print("x-shaped")
        # pprint(res["xp_shaped"])
        # print()
        # pprint(res["xp_embedded"])
        # print()
        # print("concatted")
        # pprint(res["concatted"])
        # print(res["concatted"].shape)
        # print(res["embedded"].shape)
        # print()
        # print()
        loss += res["loss"]
        accuracy_correct += res["acc_correct"]
        accuracy_total += res["acc_total"]
        batch_accuracy = res["acc_correct"] / float(res["acc_total"])
        steps += 1
        epoch_steps += 1

        if batch_id % 100 == 0:
          print("Iter {:5d} loss {:6f} accuracy {:1.2f} lr {:1.6f}".format(
            batch_id, res["loss"], batch_accuracy, lr_t))

      # evaluate on development and test set
      val_aer, val_acc = self.model.evaluate(self.dev_corpus, self.dev_wa)
      test_aer, test_acc = self.model.evaluate(self.test_corpus, self.test_wa)
      dev_AERs.append(val_aer)
      test_AERs.append(test_aer)

      # print Epoch loss
      print("Epoch {} loss {:6f} accuracy {:1.2f} val_aer {:1.2f} val_acc {:1.2f}".format(
          epoch_id,
          loss / float(epoch_steps),
          accuracy_correct / float(accuracy_total),
          val_aer, val_acc))

      # evaluate training-set likelihoods
      train_likelihood = self.likelihood(mode='train')
      train_likelihoods.append(train_likelihood)
      # evaluate dev-set likelihoods
      dev_likelihood = self.likelihood(mode='dev')
      dev_likelihoods.append(dev_likelihood)

      # save parameters
      save_path = self.model.save(self.session, path="model.ckpt")
      print("Model saved in file: %s" % save_path)

    return dev_AERs, test_AERs, train_likelihoods, dev_likelihoods


  def likelihood(self, mode='dev'):
    """
    Computes the likelihood over the entire corpus.
    Note: is this a good idea? Can we compute this?
    """
    batch_size = 1000

    if mode == 'dev':
      print('Computing dev-set likelihood')
      corpus_size = sum([1 for _ in self.dev_corpus])
      # mega_batch = list(iterate_minibatches(self.dev_corpus, batch_size=corpus_size))[0]
      batches = iterate_minibatches(self.dev_corpus, batch_size=corpus_size)
    if mode == 'train':
      print('Computing training-set likelihood')
      # corpus_size = sum([1 for _ in self.corpus])
      # mega_batch = list(iterate_minibatches(self.corpus, batch_size=corpus_size))[0]
      batches = iterate_minibatches(self.corpus, batch_size=batch_size)

    loss = 0
    for k, batch in enumerate(batches, 1):
      x, y = prepare_data(batch, self.model.x_vocabulary,
                            self.model.y_vocabulary)
        
      # Dynamic learning rate, cf. Bottou (2012), Stochastic gradient descent tricks.
      lr_t = 0

      feed_dict = {
        self.model.x:  x,
        self.model.y:  y
      }

      # things we want TF to return to us from the computation
      fetches = {
        "loss"  : self.model.loss,
      }

      res = self.session.run(fetches, feed_dict=feed_dict)

      loss += res["loss"]

    # likelihood = - loss * corpus_size
    likelihood = - loss * batch_size # now loss is

    return likelihood


