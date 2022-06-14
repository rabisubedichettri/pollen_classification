import os
from utils.config_loader import load_config,get_base_dic
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras import Model 
import cv2
import numpy as np

from model.base_model import BaseModel


class AccuracyCallbacks(tf.keras.callbacks.Callback):
    
    # def __init__(self):
    #     super().__init__(self)

    def on_epoch_end(self, epoch,logs ={}):
        # hard coded
        config_loc=os.path.join(get_base_dic(),"configs","network.json")
        config=load_config(config_loc)
        minimum_accuracy=config["custom_network"]["model_save_when_accurracy"]
        currnet_accuracy=logs.get('accuracy')
        if(currnet_accuracy >  minimum_accuracy): 
            print(f'\n {currnet_accuracy}% acc reached')
            self.model.stop_training = True

            # save the model if accuracy has been achived.
            saved_data=config["saved_data"]
            saved_dir=os.path.join(get_base_dic(),saved_data["dir_name"])
            if not os.path.exists(saved_dir):
                os.mkdir(saved_dir)
            cp_dir_r=os.path.join(saved_dir,saved_data["model"]["dir_name"])
            if not os.path.exists(cp_dir_r):
                os.mkdir(cp_dir_r)
            model_loc=os.path.join(cp_dir_r,saved_data["model"]["save"])
            self.model.save(model_loc)
        


class PollenNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.BASE_DIR=get_base_dic()
        self.load_label()
        self.data_loader()

    def _preprocess(self):
        image_augment=self.config["train"]["image_augment"]
        self.train_datagen = ImageDataGenerator(rescale = 1./image_augment["rescale"],
                                        rotation_range =image_augment["rotation_range"],
                                        width_shift_range = image_augment["width_shift_range"],
                                        height_shift_range = image_augment["height_shift_range"],
                                        horizontal_flip = image_augment["horizontal_flip"],
                                        vertical_flip = image_augment["vertical_flip"],
                                        )
        self.test_datagen = ImageDataGenerator(rescale=1./self.config["test"]["image_augment"]["rescale"])
        self.valid_datagen = ImageDataGenerator(rescale=1./self.config["val"]["image_augment"]["rescale"])
        
    
    def data_loader(self):
        self._preprocess()
        target_size = (self.config["base_network"]["input_image_width"], self.config["base_network"]["input_image_height"])
        self.train_generator = self.train_datagen.flow_from_directory(
            self.config["train"]["location"],
            target_size=target_size,
            batch_size=self.config["train"]["batch_size"],
            class_mode='categorical')
        self.test_generator = self.test_datagen.flow_from_directory(
            self.config["test"]["location"],
            target_size=target_size,
            batch_size=self.config["test"]["batch_size"],
            class_mode='categorical')
        self.valid_generator = self.valid_datagen.flow_from_directory(
            self.config["val"]["location"],
            target_size=target_size,
            batch_size=self.config["val"]["batch_size"],
            class_mode='categorical')
    

    def build(self):
        base_last_layer=self.base_model.get_layer("block7b_add")
        x=Flatten()(base_last_layer.output)
        x=Dense(1024,activation="relu")(x)
        x=Dropout(0.2)(x)
        y=Dense(self.config["custom_network"]["total_class"],activation="softmax")(x)
        self.model=Model(self.base_model.input,y)
        print("build model successfully")

    def compile(self):
        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self):
        saved_data=self.config["saved_data"]
        savedm_dir=os.path.join(get_base_dic(),saved_data["dir_name"])
        if not os.path.exists(savedm_dir):
            os.mkdir(savedm_dir)
        cp_dir_r=os.path.join(savedm_dir,saved_data["checkpoint"]["dir_name"])
        if not os.path.exists(cp_dir_r):
            os.mkdir(cp_dir_r)
        checkpoint_path = os.path.join(cp_dir_r,f'{saved_data["checkpoint"]["save"]}/cp.ckpt')
        

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        
        self.model.fit(
            self.train_generator,
            epochs = self.config["custom_network"]["epochs"],
            validation_data=self.valid_generator, 
            callbacks=[cp_callback,AccuracyCallbacks()]
        )
        

    def load_model(self):
        saved_data=self.config["saved_data"]
        model_loc=os.path.join(self.BASE_DIR,saved_data["dir_name"],saved_data["model"]['dir_name'],saved_data["model"]["save"])
        if os.path.exists(model_loc):
            self.model=tf.keras.models.load_model(model_loc)
            print("model loaded successfully.")
        else:
            print("Model not found.")
            exit()

    
    def save_model(self):
        saved_data=self.config["saved_data"]
        saved_dir=os.path.join(get_base_dic(),saved_data["dir_name"])
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        cp_dir_r=os.path.join(saved_dir,saved_data["model"]["dir_name"])
        if not os.path.exists(cp_dir_r):
            os.mkdir(cp_dir_r)
        model_loc=os.path.join(cp_dir_r,saved_data["model"]["save"])
        self.model.save(model_loc)
    
    def predict(self,img_loc):
        src = cv2.imread(img_loc,1)

        # normalize
        src=src/255.0
        # resize 
        base_network=self.config["base_network"]
        img_shape=base_network["input_image_width"],base_network["input_image_height"]
        output = cv2.resize(src, dsize=img_shape)
        y = np.expand_dims(output, axis=0)
        p=self.model.predict(y)
        index=p[0].argmax()
        print(f'Predicted class: {self.label[index]} with probabilty {p[0][index]}')

    def load_label(self):
        label_loc=os.path.join(self.BASE_DIR,self.config["custom_network"]["label_file"])
        self.label=[]
        with open(label_loc) as fp:
            while True:
                line = fp.readline()
                self.label.append(line.strip())
                if not line:
                    break
            
    def save_weight(self):
        saved_data=self.config["saved_data"]
        checkpoint_path = os.path.join(self.BASE_DIR,saved_data["dir_name"],"checkpoints",f'{saved_data["checkpoint"]["save"]}\cp.ckpt')
        self.model.save_weights(checkpoint_path)
        print("weights loaded successfully")
    
        

    def load_weight(self):
        saved_data=self.config["saved_data"]
        checkpoint_path = os.path.join(self.BASE_DIR,saved_data["dir_name"],"checkpoints",f'{saved_data["checkpoint"]["save"]}\cp.ckpt')
        self.model.load_weights(checkpoint_path)
        print("weights loaded successfully")
    

    def _model_summary(self):
        print(self.model.summary())


    def train(self):
        if self.config["activity"]["new_model"]:
            self.build()
        else:
            self.load_model()
        self._model_summary()
        self.compile()
        self.fit()


    def evaluate(self):
        self.model.evaluate(self.test_generator)


