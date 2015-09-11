from util import *
class LSTM(object):
  def __init__(self, lstm_config):
    self.name_ = lstm_config.name
    num_lstms = lstm_config.num_hid
    assert num_lstms  > 0
    self.num_lstms_   = num_lstms
    self.has_input_   = lstm_config.has_input
    self.has_output_  = lstm_config.has_output
    self.input_dims_  = lstm_config.input_dims
    self.output_dims_ = lstm_config.output_dims
    self.use_relu_    = lstm_config.use_relu
    self.input_dropprob_  = lstm_config.input_dropprob
    self.output_dropprob_ = lstm_config.output_dropprob
    self.t_ = 0

    self.w_dense_  = Param((4 * num_lstms, num_lstms), lstm_config.w_dense)
    self.w_diag_   = Param((num_lstms, 3), lstm_config.w_diag)
    self.b_        = Param((4 * num_lstms, 1), lstm_config.b)
    self.param_list_ = [
      ('%s:w_dense' % self.name_, self.w_dense_),
      ('%s:w_diag'  % self.name_, self.w_diag_),
      ('%s:b'       % self.name_, self.b_),
    ]
    if self.has_input_:
      assert self.input_dims_ > 0
      self.w_input_ = Param((4 * num_lstms, self.input_dims_), lstm_config.w_input)
      self.param_list_.append(('%s:w_input' % self.name_, self.w_input_))
    if self.has_output_:
      assert self.output_dims_ > 0
      self.w_output_ = Param((self.output_dims_, num_lstms), lstm_config.w_output)
      self.param_list_.append(('%s:w_output' % self.name_, self.w_output_))
      self.b_output_ = Param((self.output_dims_, 1), lstm_config.b_output)
      self.param_list_.append(('%s:b_output' % self.name_, self.b_output_))

  def HasInputs(self):
    return self.has_input_

  def HasOutputs(self):
    return self.has_output_

  def GetParams(self):
    return self.param_list_

  def SetBatchSize(self, batch_size, seq_length):
    assert batch_size > 0
    assert seq_length > 0
    self.batch_size_  = batch_size
    self.seq_length_  = seq_length
    self.gates_  = cm.empty((4 * self.num_lstms_, batch_size * seq_length))
    self.cell_   = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.hidden_ = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.gates_deriv_  = cm.empty_like(self.gates_)
    self.cell_deriv_   = cm.empty_like(self.cell_)
    self.hidden_deriv_ = cm.empty_like(self.hidden_)

    """
    if self.has_output_ and self.output_dropprob_ > 0:
      self.output_drop_mask_ = cm.empty_like(self.hiddenbatch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_state_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_deriv_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]

    if self.has_input_ and self.input_dropprob_ > 0:
      self.input_drop_mask_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_state_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_deriv_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
    """

  def Load(self, f):
    for name, p in self.param_list_:
      p.Load(f, name)

  def Save(self, f):
    for name, p in self.param_list_:
      p.Save(f, name)

  def Fprop(self, input_frame=None, init_cell=None, init_hidden=None, output_frame=None, train=False):
    t = self.t_
    batch_size = self.batch_size_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    start = t * batch_size
    end = start + batch_size
    gates        = self.gates_.slice(start, end)
    cell_state   = self.cell_.slice(start, end)
    hidden_state = self.hidden_.slice(start, end)
    if t == 0:
      if init_cell is None:
        if input_frame is not None: 
          assert self.has_input_
          gates.add_dot(self.w_input_.GetW(), input_frame)
        gates.add_col_vec(self.b_.GetW())
        cm.lstm_fprop2_init(gates, cell_state, hidden_state, self.w_diag_.GetW())
      else:
        cell_state.add(init_cell)
        assert init_hidden is not None
        hidden_state.add(init_hidden)
    else:
      prev_start = start - batch_size
      prev_hidden_state = self.hidden_.slice(prev_start, start)
      prev_cell_state = self.cell_.slice(prev_start, start)
      if input_frame is not None: 
        assert self.has_input_
        gates.add_dot(self.w_input_.GetW(), input_frame)
      gates.add_dot(self.w_dense_.GetW(), prev_hidden_state)
      gates.add_col_vec(self.b_.GetW())
      cm.lstm_fprop2(gates, prev_cell_state, cell_state, hidden_state, self.w_diag_.GetW())

    if self.has_output_:
      assert output_frame is not None
      output_frame.add_dot(self.w_output_.GetW(), hidden_state)
      output_frame.add_col_vec(self.b_output_.GetW())
    self.t_ += 1

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_cell=None, init_hidden=None,
                   init_cell_deriv=None, init_hidden_deriv=None,
                   output_deriv=None):
    batch_size = self.batch_size_
    self.t_ -= 1

    t = self.t_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    start = t * batch_size
    end = start + batch_size
    gates        = self.gates_.slice(start, end)
    gates_deriv  = self.gates_deriv_.slice(start, end)
    cell_state   = self.cell_.slice(start, end)
    cell_deriv   = self.cell_deriv_.slice(start, end)
    hidden_state = self.hidden_.slice(start, end)
    hidden_deriv = self.hidden_deriv_.slice(start, end)
    
    if self.has_output_:
      assert output_deriv is not None  # If this lstm's output was used, it must get a deriv back.
      self.w_output_.GetdW().add_dot(output_deriv, hidden_state.T)
      self.b_output_.GetdW().add_sums(output_deriv, axis=1)
      hidden_deriv.add_dot(self.w_output_.GetW().T, output_deriv)

    if t == 0:
      if init_cell is None:
        assert self.has_input_
        cm.lstm_outp2_init(gates_deriv, cell_state, self.w_diag_.GetdW())
        cm.lstm_bprop2_init(gates, gates_deriv, cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
        self.b_.GetdW().add_sums(gates_deriv, axis=1)
        self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
      else:
        init_hidden_deriv.add(hidden_deriv)
        init_cell_deriv.add(cell_deriv)
    else:
      prev_start = start - batch_size
      prev_hidden_state = self.hidden_.slice(prev_start, start)
      prev_hidden_deriv = self.hidden_deriv_.slice(prev_start, start)
      prev_cell_state   = self.cell_.slice(prev_start, start)
      prev_cell_deriv   = self.cell_deriv_.slice(prev_start, start)
      cm.lstm_outp2(gates_deriv, prev_cell_state, cell_state, self.w_diag_.GetdW())
      cm.lstm_bprop2(gates, gates_deriv, prev_cell_state, prev_cell_deriv,
                     cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
      self.b_.GetdW().add_sums(gates_deriv, axis=1)
      self.w_dense_.GetdW().add_dot(gates_deriv, prev_hidden_state.T)
      prev_hidden_deriv.add_dot(self.w_dense_.GetW().T, gates_deriv)
      if input_frame is not None:
        assert self.has_input_
        self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
        if input_deriv is not None:
          input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)

  def GetCurrentCellState(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.cell_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentCellDeriv(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.cell_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentHiddenState(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.hidden_.slice(t * batch_size, (t+1) * batch_size)
  
  def GetCurrentHiddenDeriv(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.hidden_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def Update(self):
    self.w_dense_.Update()
    self.w_diag_.Update()
    self.b_.Update()
    if self.has_input_:
      self.w_input_.Update()
    if self.has_output_:
      self.w_output_.Update()
      self.b_output_.Update()

  def Display(self, fig=1):
    plt.figure(2*fig)
    plt.clf()
    name = ['h', 'c', 'i', 'f', 'a', 'o']
    for i in xrange(self.seq_length_):
      state = self.state_[i].asarray()
      for j in xrange(6):
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+1)
        start = j * self.num_lstms_
        end = (j+1) * self.num_lstms_
        plt.imshow(state[:, start:end])
        _, labels = plt.xticks()
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        #plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+7)
        plt.hist(state[:, start:end].flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+13)
        plt.hist(state[:, start:end].mean(axis=0).flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        plt.title('%s:%.3f' % (name[j],state[:, start:end].mean()))

    plt.draw()
    
    plt.figure(2*fig+1)
    plt.clf()
    name = ['w_dense', 'w_diag', 'b', 'w_input']
    ws = [self.w_dense_, self.w_diag_, self.b_, self.w_input_]
    l = len(ws)
    for i in xrange(l):
      w = ws[i]
      plt.subplot(1, l, i+1)
      plt.hist(w.GetW().asarray().flatten(), 100)
      _, labels = plt.xticks()
      plt.setp(labels, rotation=45)
      plt.title(name[i])
    plt.draw()

  def Reset(self):
    self.t_ = 0
    self.gates_.assign(0)
    self.gates_deriv_.assign(0)
    self.cell_.assign(0)
    self.cell_deriv_.assign(0)
    self.hidden_.assign(0)
    self.hidden_deriv_.assign(0)
    self.b_.GetdW().assign(0)
    self.w_dense_.GetdW().assign(0)
    self.w_diag_.GetdW().assign(0)
    if self.has_input_:
      self.w_input_.GetdW().assign(0)
    if self.has_output_:
      self.w_output_.GetdW().assign(0)
      self.b_output_.GetdW().assign(0)

  def GetInputDims(self):
    return self.input_dims_
  
  def GetOutputDims(self):
    return self.output_dims_

  def GetAllStates(self):
    return self.state_

class LSTMStack(object):
  def __init__(self):
    self.models_ = []
    self.num_models_ = 0

  def Add(self, model):
    self.models_.append(model)
    self.num_models_ += 1

  def Fprop(self, input_frame=None, init_cell=[], init_hidden=[], output_frame=None, train=False):
    num_models = self.num_models_
    num_init_cell = len(init_cell)
    assert num_init_cell == 0 or num_init_cell == num_models
    assert len(init_hidden) == num_init_cell
    for m, model in enumerate(self.models_):
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_init_cell    = init_cell[m] if num_init_cell > 0 else None
      this_init_hidden  = init_hidden[m] if num_init_cell > 0 else None
      this_output_frame = output_frame if m == num_models - 1 else None
      model.Fprop(input_frame=this_input_frame,
                  init_cell=this_init_cell,
                  init_hidden=this_init_hidden,
                  output_frame=this_output_frame,
                  train=train)

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_cell=[], init_hidden=[], init_cell_deriv=[],
                   init_hidden_deriv=[], output_deriv=None):
    num_models = self.num_models_
    num_init_cell = len(init_cell)
    assert num_init_cell == 0 or num_init_cell == num_models
    assert len(init_hidden) == num_init_cell
    assert len(init_cell_deriv) == num_init_cell
    assert len(init_hidden_deriv) == num_init_cell
    for m in xrange(num_models-1, -1, -1):
      model = self.models_[m]
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_input_deriv  = input_deriv if m == 0 else self.models_[m-1].GetCurrentHiddenDeriv() 
      this_init_cell    = init_cell[m] if num_init_cell > 0 else None
      this_init_cell_deriv = init_cell_deriv[m] if num_init_cell > 0 else None
      this_init_hidden   = init_hidden[m] if num_init_cell > 0 else None
      this_init_hidden_deriv   = init_hidden_deriv[m] if num_init_cell > 0 else None
      this_output_deriv = output_deriv if m == num_models - 1 else None
      model.BpropAndOutp(input_frame=this_input_frame,
                         input_deriv=this_input_deriv,
                         init_cell=this_init_cell,
                         init_cell_deriv=this_init_cell_deriv,
                         init_hidden=this_init_hidden,
                         init_hidden_deriv=this_init_hidden_deriv,
                         output_deriv=this_output_deriv)

  def Reset(self):
    for model in self.models_:
      model.Reset()

  def Update(self):
    for model in self.models_:
      model.Update()

  def GetNumModels(self):
    return self.num_models_
  
  def SetBatchSize(self, batch_size, seq_length):
    for model in self.models_:
      model.SetBatchSize(batch_size, seq_length)

  def Save(self, f):
    for model in self.models_:
      model.Save(f)

  def Load(self, f):
    for model in self.models_:
      model.Load(f)

  def Display(self):
    for m, model in enumerate(self.models_):
      model.Display(m)

  def GetParams(self):
    params_list = []
    for model in self.models_:
      params_list.extend(model.GetParams())
    return params_list

  def HasInputs(self):
    if self.num_models_ > 0:
      return self.models_[0].HasInputs()
    else:
      return False
  
  def HasOutputs(self):
    if self.num_models_ > 0:
      return self.models_[-1].HasOutputs()
    else:
      return False

  def GetInputDims(self):
    if self.num_models_ > 0:
      return self.models_[0].GetInputDims()
    else:
      return 0
  
  def GetOutputDims(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetOutputDims()
    else:
      return 0

  def GetAllCurrentCellStates(self):
    return [m.GetCurrentCellState() for m in self.models_]
  
  def GetAllCurrentHiddenStates(self):
    return [m.GetCurrentHiddenState() for m in self.models_]
  
  def GetAllCurrentCellDerivs(self):
    return [m.GetCurrentCellDeriv() for m in self.models_]
  
  def GetAllCurrentHiddenDerivs(self):
    return [m.GetCurrentHiddenDeriv() for m in self.models_]
