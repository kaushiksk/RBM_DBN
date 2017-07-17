import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import json
import pickle



class DBN():

  def __init__(
    self,
    train_data,
    targets, 
    layers,
    outputs,
    rbm_lr,
    rbm_iters,
    rbm_dir=None,
    test_data = None,
    test_targets = None,    
    epochs = 25,
    fine_tune_batch_size = 32,
    outdir="tmp/",
    logdir="logs/"

     ):

    self.hidden_sizes = layers
    self.outputs = outputs
    self.targets = targets
    self.data = train_data

    if test_data is None:
      self.validate = False
    else:
      self.validate = True

    self.valid_data = test_data
    self.valid_labels = test_targets

    self.rbm_learning_rate = rbm_lr
    self.rbm_iters = rbm_iters

    self.epochs = epochs
    self.nn_batch_size = fine_tune_batch_size

    self.rbm_weights = []
    self.rbm_biases = []
    self.rbm_h_act = []

    self.model = None
    self.history = None

    if not os.path.exists(outdir):
      os.makedirs(outdir)
    if not os.path.exists(logdir):
      os.makedirs(logdir)


    if outdir[-1]!='/':
      outdir = outdir + '/'

    self.outdir = outdir
    self.logdir=logdir

  def pretrain(self,save=True):
    
    visual_layer = self.data

    for i in range(len(self.hidden_sizes)):
      print("[DBN] Layer {} Pre-Training".format(i+1))

      rbm = BernoulliRBM(n_components = self.hidden_sizes[i], n_iter = self.rbm_iters[i], learning_rate = self.rbm_learning_rate[i],  verbose = True, batch_size = 32)
      rbm.fit(visual_layer)
      self.rbm_weights.append(rbm.components_)
      self.rbm_biases.append(rbm.intercept_hidden_)
      self.rbm_h_act.append(rbm.transform(visual_layer))

      visual_layer = self.rbm_h_act[-1]

    if save:
      with open(self.outdir + "rbm_weights.p", 'wb') as f:
        pickle.dump(self.rbm_weights, f)

      with open(self.outdir + "rbm_biases.p", 'wb') as f:
        pickle.dump(self.rbm_biases, f)

      with open(self.outdir + "rbm_hidden.p", 'wb') as f:
        pickle.dump(self.rbm_h_act, f) 




  def finetune(self):
    model = Sequential()
    for i in range(len(self.hidden_sizes)):

      if i==0:
        model.add(Dense(self.hidden_sizes[i], activation='relu', input_dim=self.data.shape[1], name='rbm_{}'.format(i)))
      else:
        model.add(Dense(self.hidden_sizes[i], activation='relu', name='rbm_{}'.format(i)))


    model.add(Dense(self.outputs, activation='softmax'))
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    for i in range(len(self.hidden_sizes)):
      layer = model.get_layer('rbm_{}'.format(i))
      layer.set_weights([self.rbm_weights[i].transpose(),self.rbm_biases[i]])

    checkpointer = ModelCheckpoint(filepath= self.outdir + "dbn_weights.hdf5", verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=self.logdir)

    if self.validate:
      self.history = model.fit(trainx, trainy, 
                              epochs = self.epochs, 
                              batch_size = self.nn_batch_size,
                              validation_data=(self.valid_data, self.valid_labels),
                              callbacks=[checkpointer, tensorboard])
    else:
       self.history = model.fit(trainx, trainy, 
                              epochs = self.epochs, 
                              batch_size = self.nn_batch_size,
                              callbacks=[checkpointer, tensorboard])     
    self.model = model

  def report(self, data, labels):
    print(classification_report(np.argmax(labels, axis=1), np.argmax(self.model.predict(data),axis=1)))


  def save_model(self,filename):

    if self.model is None :
      raise ValueError("Run finetune() first")

    with open(self.outdir + filename, mode='w', encoding='utf-8') as outfile:

      data = {
              "model_config":self.model.get_config(),
              "loss_acc": self.history.history
          }
      json.dump(data, outfile, indent=2)

  def load_rbm(self):
    try:
      self.rbm_weights = pickle.load(self.rbm_dir + "rbm_weights.p")
      self.rbm_biases = pickle.load(self.rbm_dir + "rbm_biases.p")
      self.rbm_h_act = pickle.load(self.rbm_dir + "rbm_hidden.p")
    except:
      print("No such file or directory.")


if __name__ == '__main__':

  trainx = np.load("mnist_train.npy")
  trainy= np.load("mnist_trainy.npy")
  testx = np.load("mnist_test.npy")
  testy = np.load("mnist_testy.npy")

  dbn = DBN(train_data = trainx, targets = trainy,
            #test_data = testx, test_targets = testy,
            layers = [200],
            outputs = 10,
            rbm_iters = [40],
            rbm_lr = [0.01],
            outdir = "mnistrbm/",
            logdir = "mnistrbm_logs/"
            )
  dbn.pretrain(save=True)
  dbn.finetune()
  dbn.save_model("mnist_dbn_model.json")

  print("Training Report")
  dbn.report(trainx,trainy)

  print("Testing Report")
  dbn.report(testx,testy)

