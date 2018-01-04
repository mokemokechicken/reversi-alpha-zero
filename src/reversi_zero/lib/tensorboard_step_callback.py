from keras.callbacks import Callback
import tensorflow as tf


class TensorBoardStepCallback(Callback):
    """Tensorboard basic visualizations by step.

    """

    def __init__(self, log_dir):
        super().__init__()
        self.step = 0
        self.writer = tf.summary.FileWriter(log_dir)

    def on_batch_end(self, batch, logs=None):
        self.step += 1

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.step)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()
