import tensorflow as tf


class Forecaster:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_one(self, dataframe_row):
        prediction = self.model.predict(dataframe_row)

    def predict(self, dataframe):
        predictions = self.model.predict(dataframe)
        for p in predictions:
            yield float(p)
