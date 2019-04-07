import tensorflow as tf


def clone_with_weights(model):
    """
    returns a new keras model with exactly the same architecture and weights as the argument.
    """
    new = tf.keras.models.clone_model(model)
    new.set_weights(model.get_weights())
    return new

def track_model(*, model, tracker, lr = 0.01):
    """
    Causes the keras model "copy" to slowly approach the weights of "target".
    """
    assert 0 < lr < 1
    new_weights = [lr * w + (1-lr) * w_old for w, w_old in zip(model.get_weights(), tracker.get_weights())]
    tracker.set_weights(new_weights)