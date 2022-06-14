
import os
from utils.config_loader import load_config,get_base_dic

config_loc=os.path.join(get_base_dic(),"configs","network.json")
config=load_config(config_loc)
if not config["hardware"]["GPU"]["active"]:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from model.custom_model import PollenNet


obj=PollenNet(config)

# training from starting
obj.train()

# resume the model with saved model
# obj.load_model()
# obj.train()
obj.save_model()

#testing
obj.evaluate()


#predict
# obj.predict("dataset/train/ABBOTTS BABBLER/001.jpg")

# load_weight
# obj.load_weight()
