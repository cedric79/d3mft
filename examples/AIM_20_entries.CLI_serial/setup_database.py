import laim.read_config as reader
from laim.laim_db import *

def main(): 
    input = "config.ini"
    system_param, db_param, aim_param, learn_param = reader.get_config(input)
    gen_db(system_param, db_param)
    
if __name__ == "__main__":
    main()
