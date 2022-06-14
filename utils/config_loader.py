import json
import os
def load_config(config_file):
    try:
        config = json.load(open(config_file,"r"))
        return config
    except:
        print(f'error while loading the file {config_file}')
        exit()

def get_base_dic():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return BASE_DIR
