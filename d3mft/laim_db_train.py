from laim.ml.nn import *
from .read_config import get_config
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train_model(system_param, learn_param):
    """
    Trains a model of interest
    """

    # Train a neural network
    net = NN(system_param, learn_param)
    net.train()
    
if __name__ == "__main__":

    input = "config.ini"
    system_param, db_param, aim_param, learn_param = get_config(input)
    train_model(system_param, learn_param)
