import numpy as np
import tensorflow as tf
import os

class RNN:
    
    def __init__(self, input_size=1, output_size=1, rnn_size=128, num_layers=1,
                 num_steps=1, keep_prob=0.8, batch_size=64, init_learning_rate=0.5,
                 learning_rate_decay=0.99, init_epoch=5, max_epoch=100, MODEL_DIR = None, name = 'rnn_default'):
        self.input_size = input_size 
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.keep_prob = keep_prob    
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.init_epoch = init_epoch
        self.max_epoch = max_epoch
        self.name = name
        self.MODEL_DIR = MODEL_DIR


    """ BUILD GRAPH """
    def build_rnn_graph_with_config(self):
        tf.reset_default_graph()
        self.rnn_graph = tf.Graph()
    
        with self.rnn_graph.as_default():
    
            learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
    
            inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
            targets = tf.placeholder(tf.float32, [None, self.output_size], name="targets")
    
            def _create_one_cell():
                rnn_cell = tf.contrib.rnn.BasicRNNCell(self.rnn_size)
                if self.keep_prob < 1.0:
                    rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob = self.keep_prob)
                return rnn_cell
    
            cell = tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(self.num_layers)],
                state_is_tuple=True
            ) if self.num_layers > 1 else _create_one_cell()
    
            val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="rnn")
    
            # Before transpose, val.get_shape() = (batch_size, num_steps, rnn_size)
            # After transpose, val.get_shape() = (num_steps, batch_size, rnn_size)
            val = tf.transpose(val, [1, 0, 2])
    
            with tf.name_scope("output_layer"):
                
                last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_rnn_output")
    
                weight = tf.Variable(tf.truncated_normal([self.rnn_size, self.output_size]), name="weights")
                bias = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="biases")
                prediction = tf.add(tf.matmul(last, weight), bias, name="prediction")
    
                tf.summary.histogram("last_rnn_output", last)
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)
    
            with tf.name_scope("train"):
                # loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
                loss = tf.reduce_mean(tf.square(prediction - targets), name="loss_mse")
                optimizer = tf.train.AdamOptimizer(learning_rate)
                minimize = optimizer.minimize(loss, name="loss_mse_adam_minimize")
                tf.summary.scalar("loss_mse", loss)
    
            # Operators to use after restoring the model
            for op in [prediction, loss]:
                tf.add_to_collection('ops_to_restore', op)
    
        return self.rnn_graph


    """ TRAIN GRAPH """    
    def train_rnn_graph(self, train_X, train_y, test_X, test_y):
        """
        name (str)
        rnn_graph (tf.Graph)
        """
        def batches(x,y, batchsize):
            for i in range(0, x.shape[0], batchsize):
                yield x[i:i+batchsize], y[i:i+batchsize]
    
        final_prediction = []
        final_loss = None
    
        self.graph_name = "%s_lr%.2f_lr_decay%.3f_lstm%d_step%d_input%d_batch%d_epoch%d_kp%.3f_layer%d" % (
            self.name,
            self.init_learning_rate, self.learning_rate_decay,
            self.lstm_size, self.num_steps,
            self.input_size, self.batch_size, self.max_epoch, self.keep_prob, self.num_layers)
    
        print("Graph Name:", self.graph_name)
    
        learning_rates_to_use = RNN._compute_learning_rates(self)
        with tf.Session(graph = self.rnn_graph) as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('_logs/' + self.graph_name, sess.graph)
            writer.add_graph(sess.graph)
    
            graph = tf.get_default_graph()
            tf.global_variables_initializer().run()
    
            inputs = graph.get_tensor_by_name('inputs:0')
            targets = graph.get_tensor_by_name('targets:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')
    
            test_data_feed = {
                inputs: test_X,
                targets: test_y,
                learning_rate: 0.0
            }
    
            loss = graph.get_tensor_by_name('train/loss_mse:0')
            minimize = graph.get_operation_by_name('train/loss_mse_adam_minimize')
            prediction = graph.get_tensor_by_name('output_layer/prediction:0')
            

            for epoch_step in range(self.max_epoch):
                current_lr = learning_rates_to_use[epoch_step]
    
                for batch_X, batch_y in list(batches(train_X, train_y, self.batch_size)):
                    train_data_feed = {
                        inputs: batch_X,
                        targets: batch_y,
                        learning_rate: current_lr
                    }
                    train_loss, _ = sess.run([loss, minimize], train_data_feed)
    
                if epoch_step % 20 == 0:
                    test_loss, _pred, _summary = sess.run([loss, prediction, merged_summary], test_data_feed)
                    print("Epoch %d [%f]:" % (epoch_step, current_lr), test_loss)
    
                writer.add_summary(_summary, global_step=epoch_step)
    
            print("Final Results:")
            final_prediction, final_loss = sess.run([prediction, loss], test_data_feed)
            print(final_loss)
    
            graph_saver_dir = os.path.join(self.MODEL_DIR, self.graph_name)
            if not os.path.exists(graph_saver_dir):
                os.mkdir(graph_saver_dir)
    
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(
                graph_saver_dir, "rnn_model_%s.ckpt" % self.name), global_step=epoch_step)
            
        np.savetxt('{}.csv'.format(os.path.join(
                graph_saver_dir, "rnn_%s.ckpt" % self.name)),final_prediction)
  
          
    """ PREDICT """         
    def prediction_by_trained_graph(self, max_epoch, test_X, test_y):
        test_prediction = None
        test_loss = None
    
        with tf.Session() as sess:
            # Load meta graph
            graph_meta_path = os.path.join(
                self.MODEL_DIR, self.graph_name,
                'rnn_model_{0}.ckpt-{1}.meta'.format(self.name, max_epoch-1))
            checkpoint_path = os.path.join(self.MODEL_DIR, self.graph_name)
    
            saver = tf.train.import_meta_graph(graph_meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    
            graph = tf.get_default_graph()
    
            test_feed_dict = {
                graph.get_tensor_by_name('inputs:0'): test_X,
                graph.get_tensor_by_name('targets:0'): test_y,
                graph.get_tensor_by_name('learning_rate:0'): 0.0
            }
    
            prediction = graph.get_tensor_by_name('output_layer/prediction:0')
            loss = graph.get_tensor_by_name('train/loss_mse:0')
            test_prediction, test_loss = sess.run([prediction, loss], test_feed_dict)    
        return test_prediction, test_loss
    

    """ LEARNING RATES TO USE """         
    def _compute_learning_rates(self):
        learning_rates_to_use = [
        self.init_learning_rate * (
            self.learning_rate_decay ** max(float(i + 1 - self.init_epoch), 0.0)
        ) for i in range(self.max_epoch)
        ]
        print("Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2])
        return learning_rates_to_use
    
    
    def to_dict(self):
            dct = self.__class__.__dict__
            return {k: v for k, v in dct.iteritems() if not k.startswith('__') and not callable(v)}
    
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())
    
        
    

