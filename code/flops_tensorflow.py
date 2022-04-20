with tf.Session() as sess:
    self._log_string('**** Evluation New Result ****')
    self._log_string('Assign From checkpoint: %s'%self.last_eval_model_path)

    self.saver.restore(sess, self.last_eval_model_path)
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print(flops_to_string(flops.total_float_ops))

    reader = pywrap_tensorflow.NewCheckpointReader(self.last_eval_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    total_parameters = 0
    for key in var_to_shape_map:#list the keys of the model
         shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
         shape = list(shape)
         variable_parameters = 1
         for dim in shape:
             variable_parameters *= dim
         total_parameters += variable_parameters
    print(params_to_string(total_parameters))

def params_to_string(params_num):
    """converting number to string

    :param float params_num: number
    :returns str: number

    >>> params_to_string(1e9)
    '1000.0 M'
    >>> params_to_string(2e5)
    '200.0 k'
    >>> params_to_string(3e-9)
    '3e-09'
    """
    if params_num // 10**6 > 0:
        return str(round(params_num / 10**6, 2)) + ' M'
    elif params_num // 10**3:
        return str(round(params_num / 10**3, 2)) + ' k'
    else:
        return str(params_num)

def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'
