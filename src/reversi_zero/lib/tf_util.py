def set_session_config(per_process_gpu_memory_fraction=None, allow_soft_placement=None):
    """

    :param float per_process_gpu_memory_fraction: GPUのメモリ使用率を0~1で指定
    :param bool allow_soft_placement:  CPU上にメモリを確保するか？
    :return:
    """
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_soft_placement=allow_soft_placement,
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
