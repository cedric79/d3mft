import laim.read_config as reader
import laim.laim_db_solve as AIM_solver

def main(): 
    input = "config.ini"
    system_param, db_param, aim_param, learn_param = reader.get_config(input)
    AIM_solver.solve_AIM(system_param, db_param, aim_param)
    
if __name__ == "__main__":
    main()
