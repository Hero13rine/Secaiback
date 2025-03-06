import tensorflow as tf
from art.estimators.classification import TensorFlowClassifier

from ..base_estimator import BaseEstimator
from estimator.estimator_factory import EstimatorFactory


@EstimatorFactory.register(framework="tensorflow", task="classification")
class TensorFlowClassificationWrapper(BaseEstimator):
    """自动处理TensorFlow特殊需求"""

    def __init__(self,
                 model: tf.keras.Model,
                 clip_values=(0, 1),
                 input_shape=(224, 224, 3)):
        # 构建服务签名
        input_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)

        @tf.function(input_signature=[input_spec])
        def serving_fn(inputs):
            return {"predictions": model(inputs)}

        # 初始化ART分类器
        self.core = TensorFlowClassifier(
            model=serving_fn,
            clip_values=clip_values
        )

    def predict(self, x):
        # 自动进行NHWC转换
        if x.shape[-1] != self.core.input_shape[-1]:
            x = np.transpose(x, (0, 2, 3, 1))
        return self.core.predict(x)

    def get_core(self):
        return self.core
