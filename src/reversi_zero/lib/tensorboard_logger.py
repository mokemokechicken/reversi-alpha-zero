import tensorflow as tf


class TensorBoardLogger:
    def __init__(self, log_dir, filename_suffix=None):
        self.writer = tf.summary.FileWriter(log_dir, filename_suffix=filename_suffix)

    def log_scaler(self, info: dict, step):
        """

        :param dict info: dict of {<tag>: <value>}
        :param int step:
        :return:
        """
        for tag, value in info.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
        self.writer.flush()

