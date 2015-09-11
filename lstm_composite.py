from data_handler import *
import lstm
from util import *

class LSTMComp(object):
  def __init__(self, model):
    self.model_ = model
    self.lstm_stack_enc_ = lstm.LSTMStack()
    self.lstm_stack_dec_ = lstm.LSTMStack()
    self.lstm_stack_fut_ = lstm.LSTMStack()
    for l in model.lstm:
      self.lstm_stack_enc_.Add(lstm.LSTM(l))
    if model.dec_seq_length > 0:
      for l in model.lstm_dec:
        self.lstm_stack_dec_.Add(lstm.LSTM(l))
    if model.future_seq_length > 0:
      for l in model.lstm_future:
        self.lstm_stack_fut_.Add(lstm.LSTM(l))
    assert model.dec_seq_length > 0 or model.future_seq_length > 0
    self.is_conditional_dec_ = model.dec_conditional
    self.is_conditional_fut_ = model.future_conditional
    if self.is_conditional_dec_ and model.dec_seq_length > 0:
      assert self.lstm_stack_dec_.HasInputs()
    if self.is_conditional_fut_ and model.future_seq_length > 0:
      assert self.lstm_stack_fut_.HasInputs()
    self.squash_relu_ = False  #model.squash_relu
    self.squash_relu_lambda_ = 0 #model.squash_relu_lambda
    self.relu_data_ = False #model.relu_data
    self.binary_data_ = True #model.binary_data or self.squash_relu_
    
    if len(model.timestamp) > 0:
      old_st = model.timestamp[-1]
      ckpt = os.path.join(model.checkpoint_dir, '%s_%s.h5' % (model.name, old_st))
      f = h5py.File(ckpt)
      self.lstm_stack_enc_.Load(f)
      if model.dec_seq_length > 0:
        self.lstm_stack_dec_.Load(f)
      if model.future_seq_length > 0:
        self.lstm_stack_fut_.Load(f)
      f.close()

  def Reset(self):
    self.lstm_stack_enc_.Reset()
    self.lstm_stack_dec_.Reset()
    self.lstm_stack_fut_.Reset()
    if self.dec_seq_length_ > 0:
      self.v_dec_.assign(0)
      self.v_dec_deriv_.assign(0)
    if self.future_seq_length_ > 0:
      self.v_fut_.assign(0)
      self.v_fut_deriv_.assign(0)

  def Fprop(self, train=False):
    if self.squash_relu_:
      self.v_.apply_relu_squash(lambdaa=self.squash_relu_lambda_)
    self.Reset()
    batch_size = self.batch_size_
    # Fprop through encoder.
    for t in xrange(self.enc_seq_length_):
      v = self.v_.slice(t * batch_size, (t+1) * batch_size)
      self.lstm_stack_enc_.Fprop(input_frame=v, train=train)

    init_cell_states = self.lstm_stack_enc_.GetAllCurrentCellStates()
    init_hidden_states = self.lstm_stack_enc_.GetAllCurrentHiddenStates()

    # Fprop through decoder.
    if self.dec_seq_length_ > 0:
      self.lstm_stack_dec_.Fprop(init_cell=init_cell_states, init_hidden=init_hidden_states,
                                 output_frame=self.v_dec_.slice(0, batch_size), train=train)
      for t in xrange(1, self.dec_seq_length_):
        if self.is_conditional_dec_:
          if train:
            t2 = self.enc_seq_length_ - t
            input_frame = self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
          else:
            input_frame = self.v_dec_.slice((t-1) * batch_size, t * batch_size)
        else:
          input_frame = None
        self.lstm_stack_dec_.Fprop(input_frame=input_frame,
                                   output_frame=self.v_dec_.slice(t * batch_size, (t+1) * batch_size),
                                   train=train)
    
    # Fprop through future predictor.
    if self.future_seq_length_ > 0:
      self.lstm_stack_fut_.Fprop(init_cell=init_cell_states, init_hidden=init_hidden_states,
                                 output_frame=self.v_fut_.slice(0, batch_size), train=train)
      for t in xrange(1, self.future_seq_length_):
        if self.is_conditional_fut_:
          if train:
            t2 = self.enc_seq_length_ + t - 1
            input_frame=self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
          else:
            input_frame = self.v_fut_.slice((t-1) * batch_size, t * batch_size)
        else:
          input_frame = None
        self.lstm_stack_fut_.Fprop(input_frame=input_frame,
                                   output_frame=self.v_fut_.slice(t * batch_size, (t+1) * batch_size),
                                   train=train)
    if self.binary_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.apply_sigmoid()
      if self.future_seq_length_ > 0:
        self.v_fut_.apply_sigmoid()
    elif self.relu_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.lower_bound(0)
      if self.future_seq_length_ > 0:
        self.v_fut_.lower_bound(0)

  def BpropAndOutp(self):
    batch_size = self.batch_size_
    if self.binary_data_:
      pass
    elif self.relu_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_deriv_.apply_rectified_linear_deriv(self.v_dec_)
      if self.future_seq_length_ > 0:
        self.v_fut_deriv_.apply_rectified_linear_deriv(self.v_fut_)

    init_cell_states = self.lstm_stack_enc_.GetAllCurrentCellStates()
    init_hidden_states = self.lstm_stack_enc_.GetAllCurrentHiddenStates()
    init_cell_derivs = self.lstm_stack_enc_.GetAllCurrentCellDerivs()
    init_hidden_derivs = self.lstm_stack_enc_.GetAllCurrentHiddenDerivs()

    # Backprop through decoder.
    if self.dec_seq_length_ > 0 :
      for t in xrange(self.dec_seq_length_-1, 0, -1):
        if self.is_conditional_dec_:
          t2 = self.enc_seq_length_ - t
          input_frame=self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
        else:
          input_frame = None
        self.lstm_stack_dec_.BpropAndOutp(input_frame=input_frame,
                                          output_deriv=self.v_dec_deriv_.slice(t * batch_size, (t+1) * batch_size))
        
      self.lstm_stack_dec_.BpropAndOutp(init_cell=init_cell_states,
                                        init_cell_deriv=init_cell_derivs,
                                        init_hidden=init_hidden_states,
                                        init_hidden_deriv=init_hidden_derivs,
                                        output_deriv=self.v_dec_deriv_.slice(0, batch_size))

    # Backprop through future predictor.
    if self.future_seq_length_ > 0 :
      for t in xrange(self.future_seq_length_-1, 0, -1):
        if self.is_conditional_fut_:
          t2 = self.enc_seq_length_ + t - 1
          input_frame=self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
        else:
          input_frame = None
        self.lstm_stack_fut_.BpropAndOutp(input_frame=input_frame,
                                          output_deriv=self.v_fut_deriv_.slice(t * batch_size, (t+1) * batch_size))
        
      self.lstm_stack_fut_.BpropAndOutp(init_cell=init_cell_states,
                                        init_cell_deriv=init_cell_derivs,
                                        init_hidden=init_hidden_states,
                                        init_hidden_deriv=init_hidden_derivs,
                                        output_deriv=self.v_fut_deriv_.slice(0, batch_size))

    # Backprop thorough encoder.
    for t in xrange(self.enc_seq_length_-1, -1, -1):
      self.lstm_stack_enc_.BpropAndOutp(input_frame=self.v_.slice(t * batch_size, (t+1) * batch_size))

  def Update(self):
    self.lstm_stack_enc_.Update()
    self.lstm_stack_dec_.Update()
    self.lstm_stack_fut_.Update()

  def ComputeDeriv(self):
    batch_size = self.batch_size_
    for t in xrange(self.dec_seq_length_):
      t2 = self.enc_seq_length_ - t - 1
      dec = self.v_dec_.slice(t * batch_size, (t+1) * batch_size)
      v = self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
      deriv = self.v_dec_deriv_.slice(t * batch_size, (t+1) * batch_size)
      dec.subtract(v, target=deriv)
      deriv.divide(float(batch_size))

    for t in xrange(self.future_seq_length_):
      t2 = t + self.enc_seq_length_
      f = self.v_fut_.slice(t * batch_size, (t+1) * batch_size)
      v = self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
      deriv = self.v_fut_deriv_.slice(t * batch_size, (t+1) * batch_size)
      f.subtract(v, target=deriv)
      deriv.divide(float(batch_size))

  def GetLoss(self):
    batch_size = self.batch_size_
    for t in xrange(self.dec_seq_length_):
      t2 = self.enc_seq_length_ - t - 1
      assert t2 >= 0
      dec = self.v_dec_.slice(t * batch_size, (t+1) * batch_size)
      v = self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
      deriv = self.v_dec_deriv_.slice(t * batch_size, (t+1) * batch_size)
      if self.binary_data_:
        cm.cross_entropy_bernoulli(v, dec, target=deriv)
      else:
        dec.subtract(v, target=deriv)

    for t in xrange(self.future_seq_length_):
      t2 = t + self.enc_seq_length_
      f = self.v_fut_.slice(t * batch_size, (t+1) * batch_size)
      v = self.v_.slice(t2 * batch_size, (t2+1) * batch_size)
      deriv = self.v_fut_deriv_.slice(t * batch_size, (t+1) * batch_size)
      if self.binary_data_:
        cm.cross_entropy_bernoulli(v, f, target=deriv)
      else:
        f.subtract(v, target=deriv)

    loss_fut = 0
    loss_dec = 0
    if self.binary_data_:
      if self.dec_seq_length_ > 0:
        loss_dec = self.v_dec_deriv_.sum() / batch_size
      if self.future_seq_length_ > 0:
        loss_fut = self.v_fut_deriv_.sum() / batch_size
    else:
      if self.dec_seq_length_ > 0:
        loss_dec = 0.5 * (self.v_dec_deriv_.euclid_norm()**2) / batch_size
      if self.future_seq_length_ > 0:
        loss_fut = 0.5 * (self.v_fut_deriv_.euclid_norm()**2) / batch_size
    return loss_dec, loss_fut

  def Validate(self, data):
    data.Reset()
    dataset_size = data.GetDatasetSize()
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    loss_dec = 0
    loss_fut = 0
    for ii in xrange(num_batches):
      v_cpu, _ = data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.Fprop()
      this_loss_dec, this_loss_fut = self.GetLoss()
      if self.dec_seq_length_ > 0:
        loss_dec += this_loss_dec / (self.dec_seq_length_)
      if self.future_seq_length_ > 0:
        loss_fut += this_loss_fut / (self.future_seq_length_)
    loss_dec /= num_batches
    loss_fut /= num_batches
    return loss_dec, loss_fut

  def SetBatchSize(self, train_data):
    self.num_dims_ = train_data.GetDims()
    batch_size = train_data.GetBatchSize()
    seq_length = train_data.GetSeqLength()
    dec_seq_length    = self.model_.dec_seq_length
    future_seq_length = self.model_.future_seq_length
    #assert seq_length == dec_seq_length + future_seq_length

    self.batch_size_ = batch_size
    self.enc_seq_length_    = seq_length - future_seq_length
    self.dec_seq_length_    = dec_seq_length
    self.future_seq_length_ = future_seq_length
    self.lstm_stack_enc_.SetBatchSize(batch_size, self.enc_seq_length_)
    self.v_ = cm.empty((self.num_dims_, batch_size * seq_length))
    if dec_seq_length > 0:
      self.lstm_stack_dec_.SetBatchSize(batch_size, dec_seq_length)
      self.v_dec_ = cm.empty((self.num_dims_, batch_size * dec_seq_length))
      self.v_dec_deriv_ = cm.empty_like(self.v_dec_)

    if future_seq_length > 0:
      self.lstm_stack_fut_.SetBatchSize(batch_size, future_seq_length)
      self.v_fut_ = cm.empty((self.num_dims_, batch_size * future_seq_length))
      self.v_fut_deriv_ = cm.empty_like(self.v_fut_)

  def Save(self, model_file):
    sys.stdout.write(' Writing model to %s' % model_file)
    f = h5py.File(model_file, 'w')
    self.lstm_stack_enc_.Save(f)
    self.lstm_stack_dec_.Save(f)
    self.lstm_stack_fut_.Save(f)
    f.close()

  def Display(self, ii, fname):
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(self.v_.asarray()[:, :1000], interpolation="nearest")
    plt.subplot(2, 1, 2)
    plt.imshow(self.v_dec_.asarray()[:, :1000], interpolation="nearest")
    plt.title('Reconstruction %d' % ii)
    plt.draw()
    #plt.pause(0.1)
    plt.savefig(fname)

  def RunAndShow(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    if dataset_size % batch_size > 0:
      num_batches += 1
    end = False
    for ii in xrange(num_batches):
      v_cpu, _ = data.GetBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.Fprop()
      v_cpu = v_cpu.T.reshape(64, 64, -1, batch_size)
      rec = self.v_dec_.asarray().reshape(64, 64, -1, batch_size)
      fut = self.v_fut_.asarray().reshape(64, 64, -1, batch_size)
      for j in xrange(batch_size):
        ind = j + ii * batch_size
        if ind >= dataset_size:
          end = True
          break
        if output_dir is None:
          output_file = None
        else:
          #output_file = os.path.join(output_dir, "%.6d.pdf" % ind)
          output_file = os.path.join(output_dir, "%.6d.npz" % ind)
        #data.DisplayData(v_cpu, rec=rec, fut=fut, case_id=j, output_file=output_file)
        print output_file
        np.savez(output_file,
                 original=v_cpu[:, :, :, j],
                 rec=rec[:, :, :, j],
                 fut=fut[:, :, :, j])
      if end:
        break

  def ShowGates(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    if dataset_size % batch_size > 0:
      num_batches += 1
    end = False
    num_lstms = 200
    for ii in xrange(num_batches):
      v_cpu, _ = data.GetBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.Fprop()
      fut_states = [s.asarray().reshape(batch_size, 6, -1) for s in self.lstm_stack_fut_.GetAllStates()[0]]
      for j in xrange(batch_size):
        if j + ii * batch_size >= dataset_size:
          end = True
          break

        hidden_states = np.concatenate([s[j, 0, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)
        cell_states = np.concatenate([s[j, 1, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)
        input_gates = np.concatenate([s[j, 2, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)
        forget_gates = np.concatenate([s[j, 3, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)
        act_gates = np.concatenate([s[j, 4, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)
        output_gates = np.concatenate([s[j, 5, :num_lstms].reshape(-1, 1) for s in fut_states], axis=1)

        #cell_states = np.minimum(cell_states, 5)
        #cell_states = np.maximum(cell_states, -5)
        cell_states /= np.sqrt((cell_states**2).mean(axis=1)).reshape(-1, 1)
        plt.figure(1)
        plt.clf()
        plt.subplot(1, 6, 1)
        plt.imshow(input_gates, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Input Gates')
        plt.subplot(1, 6, 2)
        plt.imshow(forget_gates, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Forget Gates')
        plt.subplot(1, 6, 3)
        plt.imshow(act_gates, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Input')
        plt.subplot(1, 6, 4)
        plt.imshow(output_gates, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Output Gates')
        plt.subplot(1, 6, 5)
        plt.imshow(cell_states, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Cell States')
        plt.subplot(1, 6, 6)
        plt.imshow(hidden_states, interpolation="nearest", cmap=plt.cm.gray)
        plt.title('Output')
        plt.draw()
        plt.savefig('gate_pattern.pdf', bbox_inches='tight')
        plt.show()
      if end:
        break

  def Train(self, train_data, valid_data=None):
    # Timestamp the model that we are training.
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_file = os.path.join(self.model_.checkpoint_dir, '%s_%s' % (self.model_.name, st))
    self.model_.timestamp.append(st)
    WritePbtxt(self.model_, '%s.pbtxt' % model_file)
    print 'Model saved at %s.pbtxt' % model_file
   
    self.SetBatchSize(train_data)

    loss_dec = 0
    loss_fut = 0
    print_after = self.model_.print_after
    validate_after = self.model_.validate_after
    validate = validate_after > 0 and valid_data is not None
    save_after = self.model_.save_after
    save = save_after > 0
    display_after = self.model_.display_after
    display = display_after > 0

    for ii in xrange(1, self.model_.max_iters + 1):
      newline = False
      sys.stdout.write('\rStep %d' % ii)
      sys.stdout.flush()
      v_cpu, _ = train_data.GetBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.Fprop(train=True)

      # Compute Performance.
      this_loss_dec, this_loss_fut = self.GetLoss()
      if self.dec_seq_length_ > 0:
        loss_dec += this_loss_dec / (self.dec_seq_length_)
      if self.future_seq_length_ > 0:
        loss_fut += this_loss_fut / (self.future_seq_length_)
      self.ComputeDeriv()
      if ii % print_after == 0:
        loss_dec /= print_after
        loss_fut /= print_after
        sys.stdout.write(' Dec %.5f Fut %.5f' % (loss_dec, loss_fut))
        loss_dec = 0
        loss_fut = 0
        newline = True

      self.BpropAndOutp()
      self.Update()

      if display and ii % display_after == 0:
        pass
        #self.Display(ii, '%s_reconstruction.png' % model_file)
        #fut = self.v_fut_.asarray() if self.future_seq_length_ > 0 else None
        #rec = self.v_dec_.asarray() if self.dec_seq_length_ > 0 else None
        #train_data.DisplayData(v_cpu, rec=rec, fut=fut)
        #self.lstm_stack_enc_.Display()
        #self.lstm_stack_dec_.Display()
      if validate and ii % validate_after == 0:
        valid_loss_dec, valid_loss_fut = self.Validate(valid_data)
        sys.stdout.write(' VDec %.5f VFut %.5f' % (valid_loss_dec, valid_loss_fut))
        newline = True

      if save and ii % save_after == 0:
        self.Save('%s.h5' % model_file)
      if newline:
        sys.stdout.write('\n')

    sys.stdout.write('\n')

def main():
  model      = ReadModelProto(sys.argv[1])
  lstm_autoencoder = LSTMComp(model)
  train_data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
  if len(sys.argv) > 3:
    valid_data = ChooseDataHandler(ReadDataProto(sys.argv[3]))
  else:
    valid_data = None
  lstm_autoencoder.Train(train_data, valid_data)
  """
  lstm_autoencoder.GradCheck()
  """

if __name__ == '__main__':
  board = LockGPU()
  print 'Using board', board
  cm.CUDAMatrix.init_random(42)
  np.random.seed(42)
  main()
  FreeGPU(board)
