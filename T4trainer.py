import tensorflow as tf
import numpy as np
from utils import *
import random
import numpy as np


class VAETrainer:
  """
  Takes care of training a model with SGD.
  """
  def __init__(self, model, train_e_path, train_f_path,
               dev_e_path, dev_f_path, dev_wa,
               test_e_path, test_f_path, test_wa,
               num_epochs=5, batch_size=16, max_length=30, 
               lr=0.1, lr_decay=0.001,
               session=None, max_num=np.inf):

  # def __init__(self, model, train_e_path, train_f_path,
  #            dev_e_path, dev_f_path, dev_wa,
  #            test_e_path, test_f_path, test_wa,
  #            num_epochs=5,
  #            batch_size=16, max_length=30, lr=0.1, lr_decay=0.001, session=None,
  #            max_num=np.inf):
    """Initialize the trainer with a model."""

    self.model = model
    self.train_e_path = train_e_path
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.max_length = max_length
    self.lr = lr
    self.lr_decay = lr_decay
    self.session = session

    self._build_optimizer()
    
    # this loads the data into memory so that we can easily shuffle it
    # if this takes too much memory, shuffle the data on disk
    # and use gzip_reader directly
    self.corpus = list(filter_len(smart_reader(train_e_path, max_num=max_num),
                                  max_length=max_length))
    self.dev_corpus = list(filter_len(smart_reader(dev_e_path, max_num=max_num),
                              max_length=max_length))
    print('Size: {}'.format(len(self.corpus)))

    # # This loads the data into memory so that we can easily shuffle it.
    # # If this takes too much memory, shuffle the data on disk
    # # and use bitext_reader directly.
    # self.corpus = list(bitext_reader(
    #     smart_reader(train_e_path,max_num=max_num),
    #     smart_reader(train_f_path,max_num=max_num),
    #     max_length=max_length))
    # self.dev_corpus = list(bitext_reader(
    #     smart_reader(dev_e_path),
    #     smart_reader(dev_f_path)))
    # self.test_corpus = list(bitext_reader(
    #     smart_reader(test_e_path),
    #     smart_reader(test_f_path)))

  def _build_optimizer(self):
    """Buid the optimizer."""
    self.lr_ph = tf.placeholder(tf.float32)

    # You can use SGD here (with lr_decay > 0.0) but you might
    # run into NaN losses, so choose the lr carefully.
    # self.optimizer = tf.train.GradientDescentOptimizer(
    #   learning_rate=self.lr_ph).minimize(self.model.loss)
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.lr_ph).minimize(self.model.loss)

  def train(self):
    """Trains a model."""

    dev_ELBOs = []
    train_ELBOs = []
    steps = 0
    
    for epoch_id in range(1, self.num_epochs + 1):
      
      # shuffle data set every epoch
      random.shuffle(self.corpus)
      epoch_loss = 0.0
      epoch_steps = 0

      for batch_id, batch in enumerate(iterate_minibatches(
            self.corpus, batch_size=self.batch_size), 1):

        # Dynamic learning rate, cf. Bottou (2012),
        # Stochastic gradient descent tricks.
        lr_t = self.lr * (1 + self.lr * self.lr_decay * steps)**-1
        
        x = prepare_batch_data(batch, self.model.vocabulary)
        
        feed_dict = { 
          self.lr_ph : lr_t,
          self.model.x : x
        }

        fetches = {
          "optimizer": self.optimizer,
          "loss": self.model.loss,
          "ce": self.model.ce,
          "kl": self.model.kl,
          "acc_correct": self.model.accuracy_correct,
          "acc_total": self.model.accuracy_total,
          "accuracy": self.model.accuracy,
          "predictions": self.model.predictions
        }

        res = self.session.run(fetches, feed_dict=feed_dict)

        epoch_loss += res["loss"]
        steps += 1
        epoch_steps += 1
        
        if batch_id % 100 == 0:
          print("Iter {} loss {} ce {} kl {} acc {:1.2f} {}/{} lr {:1.6f}".format(
              steps, res["loss"], res["ce"], res["kl"], res["accuracy"],
               int(res["acc_correct"]), int(res["acc_total"]), lr_t))

      print("Epoch {} epoch_loss {}".format(
        epoch_id, epoch_loss / float(epoch_steps)))
      
      # evaluate training-set ELBOs
      train_ELBO = self.ELBO(mode='train')
      train_ELBOs.append(train_ELBO)
      # evaluate dev-set ELBOs
      dev_ELBO = self.ELBO(mode='dev')
      dev_ELBOs.append(dev_ELBO)
      
      # save parameters
      save_path = self.model.save(self.session, path="model.ckpt")
      print("Model saved in file: %s" % save_path)

    return None, None, dev_ELBOs, train_ELBOs


  def ELBO(self, mode='dev'):
      """
      Computes the ELBO over the entire corpus.
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
        x = prepare_batch_data(batch, self.model.vocabulary)

          
        # Dynamic learning rate, cf. Bottou (2012), Stochastic gradient descent tricks.
        lr_t = 0

        feed_dict = {
          self.model.x:  x,
        }

        # things we want TF to return to us from the computation
        fetches = {
          "loss"  : self.model.loss,
        }

        res = self.session.run(fetches, feed_dict=feed_dict)

        loss += res["loss"]

      # ELBO = - loss * corpus_size
      ELBO = - loss * batch_size # now loss is

      return ELBO

  # def ELBO(self, mode='dev'):
  #   """
  #   Computes the ELBO over the entire corpus.
  #   Note: is this a good idea? Can we compute this?
  #   """
  #   batch_size = 1000

  #   if mode == 'dev':
  #     print('Computing dev-set likelihood')
  #     corpus_size = sum([1 for _ in self.dev_corpus])
  #     # mega_batch = list(iterate_minibatches(self.dev_corpus, batch_size=corpus_size))[0]
  #     batches = iterate_minibatches(self.dev_corpus, batch_size=corpus_size)
  #   if mode == 'train':
  #     print('Computing training-set likelihood')
  #     # corpus_size = sum([1 for _ in self.corpus])
  #     # mega_batch = list(iterate_minibatches(self.corpus, batch_size=corpus_size))[0]
  #     batches = iterate_minibatches(self.corpus, batch_size=batch_size)

  #   loss = 0
  #   for k, batch in enumerate(batches, 1):
  #     x, y = prepare_data(batch, self.model.x_vocabulary,
  #                           self.model.y_vocabulary)
        
  #     # Dynamic learning rate, cf. Bottou (2012), Stochastic gradient descent tricks.
  #     lr_t = 0

  #     # TIM: this line removes the last column (because the last word does
  #     #      not preceed any other word) and adds a 0-column at the left end
  #     #      because the first word has no predecessor.
  #     yp = np.hstack((np.zeros((y.shape[0], 1)), y[:, : -1]))

  #     while yp.shape[1] < x.shape[1]:
  #         yp = np.hstack((yp, np.zeros((y.shape[0], 1))))

  #     yp = yp[:, : x.shape[1]]

  #     feed_dict = {
  #       self.model.x:  x,
  #       self.model.yp: yp,
  #       self.model.y:  y
  #     }

  #     # things we want TF to return to us from the computation
  #     fetches = {
  #       "loss"  : self.model.loss,
  #     }

  #     res = self.session.run(fetches, feed_dict=feed_dict)

  #     loss += res["loss"]

  #   # ELBO = - loss * corpus_size
  #   ELBO = - loss * batch_size # now loss is

  #   return ELBO