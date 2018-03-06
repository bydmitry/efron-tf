"""
TensorFlow implementation of Efron partial likelihood.

author: bydmitry & CamDavidsonPilon
date: 06.03.2018
"""
import tensorflow as tf

def efron_estimator_tf(time, censoring, prediction):
    n             = tf.shape(time)[0]
    sort_idx      = tf.nn.top_k(time, k=n, sorted=True).indices

    risk          = tf.gather(prediction, sort_idx)
    events        = tf.gather(censoring, sort_idx)
    otimes        = tf.gather(time, sort_idx)

    # Get unique failure times & Exclude zeros
    # NOTE: this assumes that falure times start from > 0 (greater than zero)
    otimes_cens   = otimes * events
    unique_ftimes = tf.boolean_mask(otimes_cens, tf.greater(otimes_cens, 0) )
    unique_ftimes = tf.unique(unique_ftimes).y
    m             = tf.shape(unique_ftimes)[0]

    # Define key variables:
    log_lik       = tf.Variable(0., dtype=tf.float32, trainable=False)

    tie_count     = tf.Variable([], dtype=tf.uint8,   trainable=False)
    tie_risk      = tf.Variable([], dtype=tf.float32, trainable=False)
    tie_hazard    = tf.Variable([], dtype=tf.float32, trainable=False)
    cum_hazard    = tf.Variable([], dtype=tf.float32, trainable=False)

    cum_sum       = tf.cumsum(tf.exp(risk))

    # Prepare for looping:
    i = tf.constant(0, tf.int32)
    def loop_cond(i, *args):
        return i < m

    def loop_1_step(i, tc, tr, th, ch):
        idx_b = tf.logical_and(
            tf.equal(otimes, unique_ftimes[i]),
            tf.equal(events, tf.ones_like(events)) )

        idx_i = tf.cast(
            tf.boolean_mask(
                tf.lin_space(0., tf.cast(n-1,tf.float32), n),
                tf.greater(tf.cast(idx_b, tf.int32),0)
            ), tf.int32 )

        tc = tf.concat([tc, [tf.reduce_sum(tf.cast(idx_b, tf.uint8))]], 0)
        tr = tf.concat([tr, [tf.reduce_sum(tf.gather(risk, idx_i))]], 0)
        th = tf.concat([th, [tf.reduce_sum(tf.gather(tf.exp(risk), idx_i))]], 0)

        idx_i = tf.cast(
            tf.boolean_mask(
                tf.lin_space(0., tf.cast(n-1,tf.float32), n),
                tf.greater(tf.cast(tf.equal(otimes, unique_ftimes[i]), tf.int32),0)
            ), tf.int32 )

        ch = tf.concat([ch, [tf.reduce_max(tf.gather( cum_sum, idx_i))]], 0)
        return i + 1, tc, tr, th, ch

    def loop_2_step(i, tc, tr, th, ch, likelihood):
        l = tf.cast(tc[i], tf.float32)
        J = tf.lin_space(0., l-1, tf.cast(l,tf.int32)) / l
        Dm = ch[i] - J * th[i]
        likelihood = likelihood + tr[i] - tf.reduce_sum(tf.log(Dm))
        return i + 1, tc, tr, th, ch, likelihood

    # Loops:
    _, tie_count, tie_risk, tie_hazard, cum_hazard = loop_1 = tf.while_loop(
        loop_cond, loop_1_step,
        loop_vars = [i, tie_count, tie_risk, tie_hazard, cum_hazard],
        shape_invariants = [i.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None])]
    )

    loop_2_out = tf.while_loop(
        loop_cond, loop_2_step,
        loop_vars = [i, tie_count, tie_risk, tie_hazard, cum_hazard, log_lik],
        shape_invariants = [i.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),log_lik.get_shape()]
    )

    log_lik = loop_2_out[-1]
    return tf.negative(log_lik)
