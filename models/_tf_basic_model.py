"""Contains a basic model for other models to inherit from. It is based on
    https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
"""

import os
import copy
import json
import tensorflow as tf
from pprint import pprint


class TFBasicModel():
    """A basic class for other Tensorflow models to inherit from."""

    def __init__(self, config):
        """Only require a configuration to initialise the model.
        Inputs:
            config: A dictionary with model configuration information."""
        # The model with the best hyper-parameters so far, is kept saved.
        # This is a mechanism to load it and override the reset of the
        # configuration.
        if config['best']:
            config.update(self.get_best_config(config['env_name']))

        # Make a `deepcopy` of the configuration before using it to avoid any
        # potential mutation.
        self.config = copy.deepcopy(config)

        if config['debug']:
            print('Config:')
            pprint(self.config)

        # Important to have a random seed for reproducible experimanets.
        # Remember to use it in your TF graph (`tf.set_random_seed()`).
        self.random_seed = self.config['random_seed']

        # All models share some basics hyper parameters, this is the section
        # where we copy them into the model.
        self.result_dir = self.config['result_dir']
        self.data_dir = self.config['data_dir']
        self.max_iter = self.config['max_iter']
        self.lr = self.config['lr']  # Learning rate.
        self.batch_size = self.config['batch_size']

        # Now the child Model needs some custom parameters.
        # To avoid any inheritance hell with the __init__ function, the model
        # will override this function completely.
        self.set_model_props()

        # Again, the child Model should provide its own build_graph function.
        self.graph = self.build_graph()

        # Any operations that should be in the graph but are common to all
        # models can be added this way, here.
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        # The specific initialisation of a model is not always common for each
        # so it is seperated from __init__.
        self.init()

        # At the end of this function, you want your model to be ready!

    def close(self):
        """Close the session."""
        self.sess.close()

    def set_model_props(self):
        # This function is here to be overriden completely.
        # When you look at your model, you want to know exactly which custom
        # options it needs.
        pass

    def get_best_config(self):
        # This function is here to be overriden completely.
        # It returns a dictionary used to update the initial configuration
        # (see __init__).
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to
        # other processes so they can independently generate random
        # configuration of the current model.
        raise Exception("The get_random_config function must be overriden by"
                        " the model.")

    def build_graph(self, graph):
        raise Exception("The build_graph function must be overriden by the"
                        " model")

    def load_data(self, data_set):
        raise Exception("The load_data function must be overriden by the"
                        " model")

    def infer(self):
        raise Exception("The infer function must be overriden by the model")

    def learn_from_epoch(self, data):
        # I like to separate the function to train per epoch and the function
        # to train globally.
        raise Exception("The learn_from_epoch function must be overriden by"
                        " the model.")

    def train(self):
        # This function is usually common to all your models, Here is an
        # example:
        data = self.load_data('train')

        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch(data)

        self.episode_id += 1

    def save(self):
        # This function is usually common to all your models, Here is an
        # example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = self.sess.run([global_step_t])
        if self.config['debug']:
            print("Saving to %s with global_step %d"
                  % (self.result_dir, global_step))
        self.saver.save(self.sess,
                        self.result_dir + '/model-ep_',
                        global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models but making
        # separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function.
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            # This requires the initialisation operation to be called 'init'.
            # I am assuming there will only be one initialisation in which
            # case I believe this happens automatically.
            init_op = self.graph.get_operation_by_name('init')
            self.sess.run(init_op)

            # Set the episode_id measuring how many times we have trained the
            # model. As we use `warm_start = True` this is analagous to
            # re-running training in Tensorflow.
            self.episode_id = 0
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
