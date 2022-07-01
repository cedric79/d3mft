import laim.read_config as reader
import laim.ml.nn as neural_net

def main(): 
    input = "config.ini"
    system_param, db_param, aim_param, learn_param = reader.get_config(input)
    net = neural_net.NN(system_param, learn_param)
    net.train()
    
if __name__ == "__main__":
    main()
