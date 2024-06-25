import os
import joblib
import mlflow
import logging
import numpy as np
from abc import ABC
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from time import gmtime, strftime
from keras.optimizers import Adam


class Model(ABC):
    def __init__(self, model_name):
        self.exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", f'exp_{model_name}')
        mlflow.set_experiment(experiment_name=self.exp_name)
        self.data_path = 'Dataset/'
        self.seed = int(os.getenv('PYTHONHASHSEED', 30))
        np.random.seed(self.seed)
        os.makedirs('models', exist_ok=True)
        self.model = None
        self.algorithm = None

    def mlflow_report(self, algorithm, model, params, metrics):
        logging.info(f'params: {params}\nmetrics: {metrics}')
        # mlflow.keras.log_model(model, algorithm)
        with mlflow.start_run(run_name=f'run_{strftime("%Y%m%d%H%M%S", gmtime())}'):
            mlflow.set_tag("mlflow.runName", self.exp_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # mlflow.pyfunc.save_model()
            mlflow.end_run()

        if algorithm == 'RF':
            joblib.dump(model, f'models/{algorithm}.pkl')
        else:
            tf.keras.models.save_model(model, f'models/{algorithm}.keras')
        return None

    def list_image_files(self):
        """
        list all files in the images_folder to create the X and target files list
        :return:
        """
        rgb_files_train = os.listdir(os.path.join(self.data_path, 'images/train/'))
        rgb_files_val = os.listdir(os.path.join(self.data_path, 'images/val/'))
        mask_files_train = os.listdir(os.path.join(self.data_path, 'masks/train/'))
        mask_files_val = os.listdir(os.path.join(self.data_path, 'masks/val/'))
        assert len(mask_files_train) == len(rgb_files_train), "The number of mask files and RGB files are different."
        assert len(mask_files_val) == len(rgb_files_val), "The number of mask files and RGB files are different."
        return rgb_files_train, mask_files_train, rgb_files_val, mask_files_val

    def append_lists_data_and_target(self, X, y, model, subset='train'):
        datasets_list = []
        targets_list = []
        for image_filename, target_filename in list(zip(X, y)):
            image = tf.keras.utils.load_img(os.path.join(self.data_path, f'images/{subset}/', image_filename))
            target = tf.keras.utils.load_img(os.path.join(self.data_path, f'masks/{subset}/', target_filename),
                                             color_mode='grayscale')
            image_array = tf.keras.utils.img_to_array(image, dtype=np.uint8)
            target_array = tf.keras.utils.img_to_array(target, dtype=bool).astype(np.uint8)
            if (np.sum(np.isnan(image_array)) == 0) and (np.sum(np.isinf(image_array)) == 0):
                if model in ['ANN', 'RF']:  # for these two models append l=as tables
                    datasets_list.append(image_array.reshape(-1, image_array.shape[-1]))
                    targets_list.append(target_array.reshape(-1, target_array.shape[-1]))
                else:
                    datasets_list.append(image_array)
                    targets_list.append(target_array)
            else:
                print("isna", image_filename)
        return datasets_list, targets_list

    def load_train_val_data(self, model, is_test=True):
        """
        TODO: use image_dataset_from_directory to get a tf.dataset.
            This will improve memory performance.
        load target and validation data from Dataset folder
        :param model:
        :return:
        """
        logging.info('reading processed images and masks.')
        X_train, y_train, X_val, y_val = self.list_image_files()
        # for test purposes, do not use all data
        if is_test:
            size_train = 10
            size_val = int(size_train * 0.2)
            X_train, y_train, X_val, y_val = (X_train[:size_train], y_train[:size_train], X_val[:size_val],
                                              y_val[:size_val])
        datasets_train, targets_train = self.append_lists_data_and_target(X_train, y_train, model, subset='train')
        datasets_val, targets_val = self.append_lists_data_and_target(X_val, y_val, model, subset='val')
        return datasets_train, targets_train, datasets_val, targets_val

    @staticmethod
    def intersection_over_union(target, prediction):
        target = tf.cast(target, dtype=tf.float32)
        prediction = tf.cast(prediction > 0.5, dtype=tf.float32)  # model outputs probability
        intersection = tf.reduce_sum(tf.multiply(target, prediction))  # logical and in tf
        union = tf.reduce_sum(target) + tf.reduce_sum(prediction) - intersection  # logical or in tf
        iou_score = intersection / union
        return iou_score

    @staticmethod
    def get_metrics_tf(model):
        return {
            "train loss": np.round(model.history.history["loss"][-1], 2),
            "validation loss": np.round(model.history.history["val_loss"][-1], 2),
            "train iou": np.round(model.history.history["intersection_over_union"][-1], 2),
            "validation iou": np.round(model.history.history["val_intersection_over_union"][-1], 2)
        }

    def load_saved_model(self, algorithm):
        logging.info('Doing predictions')
        if algorithm != 'RF':
            model = tf.keras.models.load_model(os.path.join('models', f'{algorithm}.keras'),
                                               custom_objects={'intersection_over_union': self.intersection_over_union},
                                               safe_mode=False)
        else:
            model = joblib.load(os.path.join('models', f'{algorithm}.pkl'))
        return model

    def make_predictions(self, df_validation, use_saved_model=True):
        logging.info('Doing predictions')
        if use_saved_model:
            loaded_model = self.load_saved_model(self.algorithm)
            predictions = loaded_model.predict(df_validation)
        else:
            predictions = self.model.predict(df_validation)
        predictions = (predictions > 0.5).astype(np.uint8)
        predictions = predictions[0] if self.algorithm in ['UNET', 'SEGNET'] else predictions
        return predictions


class ANN_model(Model):
    def __init__(self):
        super().__init__(model_name='ANN')
        self.algorithm = 'ANN'
        self.model = Sequential()

    def train(self, df_train, label_cat_train, df_validation, label_cat_validation):
        logging.info('Training ANN model')
        self.model.add(Dense(units=10, input_dim=df_train.shape[1], activation='relu'))
        # model.add(Dense(units=10, activation='relu'))
        self.model.add(Dense(units=8, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.intersection_over_union])
        self.model.summary()
        params = {"batch_size": 10240, "epochs": 8}  # "epochs": 8
        self.model.fit(df_train, label_cat_train, validation_data=(df_validation, label_cat_validation),
                       workers=-1, **params)
        return params

    def run(self, use_weights=None):
        if use_weights:
            logging.info(f'loading weights from {use_weights}')
            self.model.load_weights(use_weights)
        df_train, targets_train, df_val, targets_val = self.load_train_val_data(model=self.algorithm)
        df_train = np.concatenate(df_train, axis=0)
        targets_train = np.concatenate(targets_train, axis=0)
        df_val = np.concatenate(df_val, axis=0)
        targets_val = np.concatenate(targets_val, axis=0)
        params = self.train(df_train, targets_train, df_val, targets_val)
        metrics = self.get_metrics_tf(self.model)
        self.mlflow_report(self.algorithm, self.model, params, metrics)


class RF_model(Model):
    def __init__(self):
        super().__init__(model_name='RF')
        self.algorithm = 'RF'
        self.model = RandomForestClassifier(random_state=self.seed, n_jobs=-2, max_depth=5, verbose=2)

    @staticmethod
    def intersection_over_union(target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def train(self, df_train, target_train, df_val, target_val):
        logging.info('Training RF model')
        params = {"rf_params": 'default params'}
        self.model.fit(df_train, target_train)
        return params

    def get_metrics_rf(self, df_train, targets_train, df_val, targets_val):
        logging.info('Doing predictions')
        predictions_train = self.model.predict(df_train)
        predictions_validation = self.model.predict(df_val)
        iou_score_train = self.intersection_over_union(targets_train, predictions_train)
        iou_score_validation = self.intersection_over_union(targets_val, predictions_validation)
        metrics = {
            "train iou": np.round(iou_score_train, 2),
            "validation iou": np.round(iou_score_validation, 2)
        }
        return metrics

    def run(self, use_weights=None):
        if use_weights:
            logging.info('Nothing to do for RF')
        df_train, targets_train, df_val, targets_val = self.load_train_val_data(model=self.algorithm)
        df_train = np.concatenate(df_train, axis=0)
        targets_train = np.ravel(np.concatenate(targets_train, axis=0))
        df_val = np.concatenate(df_val, axis=0)
        targets_val = np.ravel(np.concatenate(targets_val, axis=0))
        params = self.train(df_train, targets_train, df_val, targets_val)
        metrics = self.get_metrics_rf(df_train, targets_train, df_val, targets_val)
        self.mlflow_report(self.algorithm, self.model, params, metrics)


class UNET_model(Model):
    def __init__(self):
        super().__init__(model_name='UNET')
        self.algorithm = 'UNET'
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        encoder_list = self.encoder(input_layer)
        output = self.decoder(encoder_list, num_classes=1)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output)
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[self.intersection_over_union])

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
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=19, monitor='val_loss'),
                     tf.keras.callbacks.TensorBoard(log_dir='logs')]
        params = {"batch_size": 10, "epochs": 30}
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), callbacks=callbacks,
                       **params)
        return params

    def run(self, use_weights=None):
        if use_weights:
            logging.info(f'loading weights from {use_weights}')
            self.model.load_weights(use_weights)
        df_train, targets_train, df_val, targets_val = self.load_train_val_data(model=self.algorithm)
        df_train = np.array(df_train)
        targets_train = np.array(targets_train)
        df_val = np.array(df_val)
        targets_val = np.array(targets_val)
        params = self.train(df_train, targets_train, df_val, targets_val)
        metrics = self.get_metrics_tf(self.model)
        self.mlflow_report(self.algorithm, self.model, params, metrics)


class SEGNET_model(Model):
    def __init__(self):
        super().__init__(model_name='SEGNET')
        self.algorithm = 'SEGNET'
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        encoder_list = self.encoder(input_layer)
        output = self.decoder(encoder_list, num_classes=1)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output)
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[self.intersection_over_union])

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
        return c5

    @staticmethod
    def decoder(encoder_output, num_classes: int):
        c5 = encoder_output

        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.BatchNormalization()(u6)
        u6 = tf.keras.layers.ReLU()(u6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
        u7 = tf.keras.layers.BatchNormalization()(u7)
        u7 = tf.keras.layers.ReLU()(u7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
        u8 = tf.keras.layers.BatchNormalization()(u8)
        u8 = tf.keras.layers.ReLU()(u8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
        u9 = tf.keras.layers.BatchNormalization()(u9)
        u9 = tf.keras.layers.ReLU()(u9)

        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)
        return outputs

    def train(self, X_train, y_train, X_validation, y_validation):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=19, monitor='val_loss'),
                     tf.keras.callbacks.TensorBoard(log_dir='logs')]
        params = {"batch_size": 2, "epochs": 30}
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), callbacks=callbacks,
                       **params)
        return params

    def run(self, use_weights=None):
        if use_weights:
            logging.info(f'loading weights from {use_weights}')
            self.model.load_weights(use_weights)
        df_train, targets_train, df_val, targets_val = self.load_train_val_data(model=self.algorithm)
        df_train = np.array(df_train)
        targets_train = np.array(targets_train)
        df_val = np.array(df_val)
        targets_val = np.array(targets_val)
        params = self.train(df_train, targets_train, df_val, targets_val)
        metrics = self.get_metrics_tf(self.model)
        self.mlflow_report(self.algorithm, self.model, params, metrics)
