import tensorflow as tf
import pdb
import numpy as np
class Seq2Seq:

    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_layers=3,output_keep_prob=0.75):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.enc_input_reverse = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.int64, [None, None])

        self.weights = tf.Variable(tf.ones([self.n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self._build_model(output_keep_prob)

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self,output_keep_prob ):
        # self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        # self.dec_input = tf.transpose(self.dec_input, [1, 0, 2])

        enc_cell, dec_cell = self._build_cells(output_keep_prob)
        enc_cell = Wrapper(enc_cell)

        with tf.variable_scope('encode_forward'):
            enc_forward_outputs, enc_states_forward_final = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('encode_backward'):
            enc_backward_outputs, enc_states_backward_final = tf.nn.dynamic_rnn(enc_cell, self.enc_input_reverse, dtype=tf.float32)

        enc_states = []
        enc_states_forward = enc_forward_outputs[0]
        enc_states_backward = enc_backward_outputs[0]


        for i, item in enumerate(enc_states_forward):
            enc_states.append(tf.contrib.rnn.LSTMStateTuple(tf.concat((item[0], enc_states_backward[i][0]), axis=2),
                                                            tf.concat((item[1], enc_states_backward[i][1]), axis=2)))

        
        ################################################################################################################
        
        # Decoder에 처음으로 들어갈 context vector의 값을 설정해 보세요
        
        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32,
                                                    initial_state=enc_states_backward_final)

        ################################################################################################################

        self.logits, self.cost, self.train_op = self._build_ops(outputs, self.targets)

        self.outputs = tf.argmax(self.logits, 2)
        
    def _cell(self, output_keep_prob, n_hidden):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def _build_cells(self, output_keep_prob=0.75):
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell(output_keep_prob, self.n_hidden)
                                                for _ in range(self.n_layers)]) # self.n_layers 만큼 deep한 RNN 네트워크 구성 (for encoder) // _cell은 LSTM wrapper임
        dec_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell(output_keep_prob, self.n_hidden)
                                                for _ in range(self.n_layers)]) # self.n_layers 만큼 deep한 RNN 네트워크 구성 (for decoder)

        return enc_cell, dec_cell

    def _build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_forward_input,enc_reverse_input, dec_input, targets):

        enc_input_reverse=[np.flip(i,axis=0) for i in enc_reverse_input]
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_forward_input,
                                      self.dec_input: dec_input,
                                      self.enc_input_reverse: enc_input_reverse,
                                      self.targets: targets})

    def test(self, session, enc_forward_input, enc_input_reverse, dec_input, targets):
        enc_input_reverse=[np.flip(i,axis=0) for i in enc_input_reverse]


        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_forward_input,
                                      self.enc_input_reverse: enc_input_reverse,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_forward_input, enc_reverse_input, dec_input):

        enc_input_reverse=[np.flip(i,axis=0) for i in enc_reverse_input]
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_forward_input,
                                      self.enc_input_reverse: enc_input_reverse,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.targets: targets})

        writer.add_summary(summary, self.global_step.eval())


class Wrapper(tf.nn.rnn_cell.RNNCell):
  def __init__(self, inner_cell):
     super(Wrapper, self).__init__()
     self._inner_cell = inner_cell

  @property
  def state_size(self):
     return self._inner_cell.state_size

  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def call(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state
