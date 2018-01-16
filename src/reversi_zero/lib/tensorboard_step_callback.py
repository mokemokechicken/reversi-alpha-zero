from keras.callbacks import Callback
import tensorflow as tf


class TensorBoardStepCallback(Callback):
    """Tensorboard basic visualizations by step.

    """

    def __init__(self, log_dir, logging_per_steps=100, step=0):
        super().__init__()
        self.step = step
        self.logging_per_steps = logging_per_steps
        self.writer = tf.summary.FileWriter(log_dir)

    def on_batch_end(self, batch, logs=None):
        self.step += 1

        if self.step % self.logging_per_steps > 0:
            return

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.step)
        self.writer.flush()

    def close(self):
        self.writer.close()
