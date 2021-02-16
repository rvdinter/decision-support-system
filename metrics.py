import tensorflow as tf
from tensorflow.python.keras.metrics import SensitivitySpecificityBase


class WorkSavedOverSamplingAtRecall(SensitivitySpecificityBase):
    """
    Work saved over sampling at %recall metric
    """

    def __init__(self, recall, num_thresholds=200, name="wss_at_recall", dtype=None):
        if recall < 0 or recall > 1:
            raise ValueError('`recall` must be in the range [0, 1].')
        self.recall = recall
        self.num_thresholds = num_thresholds
        super(WorkSavedOverSamplingAtRecall, self).__init__(
            value=recall, num_thresholds=num_thresholds, name=name, dtype=dtype
        )

    def result(self):
        recalls = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        n = self.true_negatives + self.true_positives + self.false_negatives + self.false_positives
        wss = tf.math.divide_no_nan(
            self.true_negatives + self.false_negatives, n
        )
        return self._find_max_under_constraint(
            recalls, wss, tf.math.greater_equal
        )

    def get_config(self):
        """For serialization purposes"""
        config = {'num_thresholds': self.num_thresholds, 'recall': self.recall}
        base_config = super(WorkSavedOverSamplingAtRecall, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
