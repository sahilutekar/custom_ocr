import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from modelArch import ocrArch
from model import model


class Trainer:
    def __init__(self, dataset_path, model_path, batch_size, epochs, TARGET_WIDTH, target_height, target_depth):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.TARGET_WIDTH = TARGET_WIDTH
        self.target_height = target_height
        self.target_depth = target_depth

    def train(self):
        # Set up the data generator to flow data from disk
        print("[INFO] Setting up Data Generator...")
        data_gen = ImageDataGenerator(
            validation_split=0.1, 
            rescale=1./255, 
            rotation_range=20, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            shear_range=0.2, 
            zoom_range=0.2, 
            horizontal_flip=True
        )

        train_generator = data_gen.flow_from_directory(
            self.dataset_path, 
            subset='training',
            target_size = (self.TARGET_WIDTH, self.target_height),
            batch_size = self.batch_size
        )

        val_generator = data_gen.flow_from_directory(
            self.dataset_path,
            subset='validation',
            target_size = (self.TARGET_WIDTH, self.target_height),
            batch_size = self.batch_size
        )

        # Build model
        print("[INFO] Compiling model...")
        # model = ocrArch(train_generator.num_classes, (self.TARGET_WIDTH, self.target_height, self.target_depth))
        # model = ocrArch(train_generator.num_classes, (self.TARGET_WIDTH, self.target_height))
        model1 = model(train_generator.num_classes, (TARGET_WIDTH, TARGET_HEIGHT, TARGET_DEPTH))



        # Compile the model
        # model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model1.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


        # Train the network
        print("[INFO] Training network ...")
        # Set the learning rate decay and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        H = model1.fit(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_steps = val_generator.samples // self.batch_size,
            epochs=self.epochs, verbose=1, callbacks=[reduce_lr, early_stop])

        # save the model to disk
        print("[INFO] Serializing network...")
        model1.save(self.model_path + os.path.sep + "trained_model_30.h5")

        print("[INFO] Done!")

if __name__ == "__main__":
    # Define constants
    DATASET_PATH = 'C:/Users/sahil/Desktop/lpr/alpr/char_mixed_dataset/alphaDigit'
    MODEL_PATH = '. '
    BATCH_SIZE = 128
    EPOCHS = 20
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128
    TARGET_DEPTH = 3

    # Initialize the trainer
    trainer = Trainer(DATASET_PATH, MODEL_PATH, BATCH_SIZE, EPOCHS, TARGET_WIDTH, TARGET_HEIGHT, TARGET_DEPTH) 

    # Train the model
    trainer.train()
