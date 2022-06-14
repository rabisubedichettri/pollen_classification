import logging
import os 
from config_loader import load_config,get_base_dic

class LoggerW:
    def __init__(self,to):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.config_loc=os.path.join(get_base_dic(),"configs","log.json")
        self.config=load_config(self.config_loc)
        self.logging_directory=os.path.join(get_base_dic(),self.config["log_folder_name"])
        self.log_loc=os.path.join(self.logging_directory,self.config[to])
        
        self._select_filehandler()
        formatter=logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        self.handler.setFormatter(formatter)
        
            

    def _check_filehandler(self,to):
        try:
            if not os.path.exists(self.logging_directory):
                os.mkdir(self.logging_directory)

            if not os.path.isfile(self.log_loc):
                fp = open(self.log_loc, 'w')
                fp.close()
            return True
        except:
            return False
        
        
            # return False
      
        

    def _select_filehandler(self):
        if self._check_filehandler(self.log_loc):
            self.handler=logging.FileHandler(self.log_loc)
            self.logger.addHandler(self.handler)
        else:
            print("inconsistent mapping in log enviromenr file! fix it.")
            exit()

    

    def write_debug(self,message):
        self.logger.debug(message)
    
    def write_info(self,message):
        self.logger.debug(message)

    def write_warning(self,message):
        self.logger.warning(message)
    
    
    def write_error(self,message):
        self.logger.error(message)
    
    
    def write_critical(self,message):
        self.logger.critical(message)

        
