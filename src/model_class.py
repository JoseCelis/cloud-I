import os
import mlflow
import logging
import numpy as np
import tensorflow as tf
from abc import ABC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Input, InputLayer, Flatten
from time import gmtime, strftime


class Model(ABC):
    def __init__(self):
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", f'exp_{strftime("%Y%m%d%H%M%S", gmtime())}'))
        self.data_path = 'preprocessed_data/'
        self.seed = int(os.getenv('PYTHONHASHSEED', 30))
        np.random.seed(self.seed)

    def mlflow_report(self, params, metrics):
        logging.info(f'params: {params}\nmetrics: {metrics}')
        with mlflow.start_run(run_name="experiment 1") as run:
            mlflow.set_tag("mlflow.runName", "experiment 1")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.end_run()
        return None

    def list_image_files(self):
        """
        list all files in the images_folder to create the X and target files list
        :param images_folder:
        :return:
        """
        rgb_files = [file for file in os.listdir(self.data_path) if 'RGB_' in file]
        mask_files = [file for file in os.listdir(self.data_path) if 'MASK_' in file]
        rgb_files.sort()
        mask_files.sort()
        assert len(mask_files) == len(rgb_files), "The number of mask files and RGB files are different. "
        return rgb_files, mask_files

    def append_images(self, model):
        datasets_list = []
        targets_list = []
        logging.info('reading processed images and masks.')
        X, y = self.list_image_files()
        for image_filename, target_filename in list(zip(X, y))[:10]:
        # for image_filename, target_filename in list(zip(X, y) ):
            try:
                image_array = np.load(os.path.join("preprocessed_data", image_filename))
                target_array = np.load(os.path.join("preprocessed_data", target_filename))

                if (np.sum(np.isnan(image_array)) == 0) and (np.sum(np.isinf(image_array)) == 0):
                    if model in ['ANN', 'RF']:  # for these two models append l=as tables
                        datasets_list.append(image_array.reshape(-1, image_array.shape[-1]))
                        targets_list.append(target_array.reshape(-1, target_array.shape[-1]))
                    else:
                        datasets_list.append(image_array)
                        targets_list.append(target_array)
                else:
                    print("isna", image_filename)
            except:
                print("doesn't exist", image_filename)
        return datasets_list, targets_list

    def train_validation_split(self, dataset, target_bin, fraction_val_set=0.2, model='ANN'):
        logging.info('Splitting data into train and validation')
        if model in ['ANN', 'RF']:  # for these two models append l=as tables
            df_train, df_validation, target_bin_train, target_bin_validation = train_test_split(
                dataset, target_bin, test_size=fraction_val_set, random_state=self.seed, stratify=target_bin
            )
        else:
            all_indexes = np.arange(dataset.shape[0])
            val_indexes = np.random.randint(0, dataset.shape[0], int(dataset.shape[0] * fraction_val_set))
            train_indexes = np.array(list(set(all_indexes) - set(val_indexes)))
            df_train = dataset[train_indexes, :, :, :]
            target_bin_train = target_bin[train_indexes, :, :, :]
            df_validation = dataset[val_indexes, :, :, :]
            target_bin_validation = target_bin[val_indexes, :, :, :]
        return df_train, df_validation, target_bin_train, target_bin_validation


class ANN_model(Model):
    def __init__(self):
        super().__init__()
        self.algorithm = 'ANN'
        self.model = Sequential()

    @staticmethod
    def make_target_cat(target_bin_train, target_bin_validation):
        label_cat_train = to_categorical(target_bin_train)
        label_cat_validation = to_categorical(target_bin_validation)
        return label_cat_train, label_cat_validation

    def train(self, df_train, label_cat_train, df_validation, label_cat_validation):
        logging.info('Training ANN model')
        self.model.add(Dense(units=10, input_dim=df_train.shape[1], activation='relu'))
        # model.add(Dense(units=10, activation='relu'))
        self.model.add(Dense(units=8, activation='relu'))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.fit(df_train, label_cat_train, validation_data=(df_validation, label_cat_validation),
                       batch_size=256, epochs=8, workers=-1)
        return None

    def make_predictions(self, df_validation):
        logging.info('Doing predictions')
        predictions = self.model.predict(df_validation)
        return predictions

    def run(self):
        datasets_list, targets_list = self.append_images(model='ANN')
        dataset = np.concatenate(datasets_list, axis=0)
        target = np.concatenate(targets_list, axis=0)
        df_train, df_validation, target_bin_train, target_bin_validation = self.train_validation_split(
            dataset, target, model=self.algorithm)
        label_cat_train, label_cat_validation = self.make_target_cat(target_bin_train, target_bin_validation)
        self.train(df_train, label_cat_train, df_validation, label_cat_validation)
        self.make_predictions(df_validation)


class RF_model(Model):
    def __init__(self):
        super().__init__()
        self.algorithm = 'RF'
        self.model = RandomForestClassifier(random_state=self.seed, n_jobs=-2)

    def train(self, df_train, target_bin_train, df_validation, target_bin_validation):
        logging.info('Training RF model')
        self.model.fit(df_train, target_bin_train)
        return None

    def make_predictions(self, df_validation):
        logging.info('Doing predictions')
        predictions = self.model.predict(df_validation)
        return predictions

    def run(self):
        datasets_list, targets_list = self.append_images(model=self.algorithm)
        dataset = np.concatenate(datasets_list, axis=0)
        target = np.concatenate(targets_list, axis=0)
        df_train, df_validation, target_bin_train, target_bin_validation = self.train_validation_split(
            dataset, target, model=self.algorithm)
        self.train(df_train, target_bin_train, df_validation, target_bin_validation)
        self.make_predictions(df_validation)


class UNET_model(Model):
    def __init__(self):
        super().__init__()
        self.algorithm = 'UNET'
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        encoder_list = self.encoder(input_layer)
        output = self.decoder(encoder_list, num_classes=1)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output)

    @staticmethod
    def encoder(encoder_input):
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            encoder_input)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        b1 = tf.keras.layers.BatchNormalization()(c1)
        r1 = tf.keras.layers.ReLU()(b1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        b2 = tf.keras.layers.BatchNormalization()(c2)
        r2 = tf.keras.layers.ReLU()(b2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        b3 = tf.keras.layers.BatchNormalization()(c3)
        r3 = tf.keras.layers.ReLU()(b3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        b4 = tf.keras.layers.BatchNormalization()(c4)
        r4 = tf.keras.layers.ReLU()(b4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        b5 = tf.keras.layers.BatchNormalization()(c5)
        r5 = tf.keras.layers.ReLU()(b5)
        c5 = tf.keras.layers.Dropout(0.3)(r5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        return [c5, c4, c3, c2, c1]

    @staticmethod
    def decoder(encoder_output: list, num_classes: int):
        c5 = encoder_output[0]
        c4 = encoder_output[1]
        c3 = encoder_output[2]
        c2 = encoder_output[3]
        c1 = encoder_output[4]

        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.BatchNormalization()(u6)
        u6 = tf.keras.layers.ReLU()(u6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.BatchNormalization()(u7)
        u7 = tf.keras.layers.ReLU()(u7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.BatchNormalization()(u8)
        u8 = tf.keras.layers.ReLU()(u8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        u9 = tf.keras.layers.BatchNormalization()(u9)
        u9 = tf.keras.layers.ReLU()(u9)

        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)
        return outputs

    def train(self, X_train, y_train, X_validation, y_validation):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
                     tf.keras.callbacks.TensorBoard(log_dir='logs')]
        params = {"batch_size": 32, "epochs": 2}
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), callbacks=callbacks,
                       **params)
        return params

    def make_predictions(self, df_validation):
        logging.info('Doing predictions')
        predictions = self.model.predict(df_validation)
        return predictions

    def run(self):
        datasets_list, targets_list = self.append_images(model=self.algorithm)
        dataset = np.array(datasets_list)
        target = np.array(targets_list)
        df_train, df_validation, target_bin_train, target_bin_validation = self.train_validation_split(
            dataset, target, model=self.algorithm)
        params = self.train(df_train, target_bin_train, df_validation, target_bin_validation)
        # self.make_predictions(df_validation)
        metrics = {"train loss": np.round(self.model.history.history["loss"][-1], 2),
                   "validation loss": np.round(self.model.history.history["val_loss"][-1], 2)}
        self.mlflow_report(params, metrics)


class FCN_model(Model):
    def __init__(self):
        super().__init__()
        self.algorithm = 'FCN'
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        encoder_list = self.encoder(input_layer)
        output = self.decoder(encoder_list, num_classes=1)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output)

    @staticmethod
    def encoder(encoder_input):
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            encoder_input)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)

        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
        c6 = tf.keras.layers.Dropout(0.3)(c6)
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        return c6

    @staticmethod
    def decoder(encoder_output, num_classes):
        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(encoder_output)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        # Adding the result of the deconvolution and convolution layers improves the segmentation detail.
        c7 = tf.keras.layers.Add()([u7, c7])  # Returns the sum of layers

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Add()([u8, c8])

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Add()([u9, c9])

        u10 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c9)
        c10 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = tf.keras.layers.Add()([u10, c10])

        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c10)
        return outputs

    def train(self, X_train, y_train, X_validation, y_validation):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
                     tf.keras.callbacks.TensorBoard(log_dir='logs')]
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                       batch_size=32, epochs=8, callbacks=callbacks)

    def make_predictions(self, df_validation):
        logging.info('Doing predictions')
        predictions = self.model.predict(df_validation)
        return predictions

    def run(self):
        datasets_list, targets_list = self.append_images(model=self.algorithm)
        dataset = np.array(datasets_list)
        target = np.array(targets_list)
        df_train, df_validation, target_bin_train, target_bin_validation = self.train_validation_split(
            dataset, target, model=self.algorithm)
        self.train(df_train, target_bin_train, df_validation, target_bin_validation)
        self.make_predictions(df_validation)
