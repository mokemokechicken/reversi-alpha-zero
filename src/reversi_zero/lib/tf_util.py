def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None):
    """

    :param allow_growth: 必要になったらGPUメモリを確保する
    :param float per_process_gpu_memory_fraction: GPUのメモリ使用率を0~1で指定

    :return:
    """
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
