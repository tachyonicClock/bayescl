# https://github.com/aviralkumar2907/MMCE/blob/master/20ng_mmce.py#L151
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import torch
from torch import Tensor
import torch.nn.functional as F
from hypothesis import given, strategies as st

def tf_get_out_tensor(tensor1, tensor2):
    return tf.reduce_mean(tensor1 * tensor2)


def tf_calibration_mmce_w_loss(logits, correct_labels):
    """Function to compute the MMCE_w loss."""
    predicted_probs = tf.nn.softmax(logits)
    range_index = tf.cast(
        tf.expand_dims(tf.range(0, tf.shape(predicted_probs)[0]), 1), tf.int64
    )
    predicted_labels = tf.argmax(predicted_probs, axis=1)
    gather_index = tf.concat([range_index, tf.expand_dims(predicted_labels, 1)], axis=1)
    predicted_probs = tf.reduce_max(predicted_probs, 1)
    correct_mask = tf.where(
        tf.equal(correct_labels, predicted_labels),
        tf.ones(tf.shape(correct_labels)),
        tf.zeros(tf.shape(correct_labels)),
    )
    sigma = 0.2

    def tf_kernel(matrix):
        """Kernel was taken to be a laplacian kernel with sigma = 0.4."""
        return tf.exp(-1.0 * tf.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (2 * 0.2))

    k = tf.cast(tf.reduce_sum(correct_mask), tf.int32)
    k_p = tf.cast(tf.reduce_sum(1.0 - correct_mask), tf.int32)
    cond_k = tf.where(tf.equal(k, 0), 0, 1)
    cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)
    k = tf.maximum(k, 1) * cond_k * cond_k_p + (1 - cond_k * cond_k_p) * 2
    k_p = tf.maximum(k_p, 1) * cond_k_p * cond_k + (
        (1 - cond_k_p * cond_k) * (tf.shape(correct_mask)[0] - 2)
    )
    correct_prob, _ = tf.nn.top_k(predicted_probs * correct_mask, k)
    incorrect_prob, _ = tf.nn.top_k(predicted_probs * (1 - correct_mask), k_p)

    def get_pairs(tensor1, tensor2):
        correct_prob_tiled = tf.expand_dims(
            tf.tile(tf.expand_dims(tensor1, 1), [1, tf.shape(tensor1)[0]]), 2
        )
        incorrect_prob_tiled = tf.expand_dims(
            tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor2)[0]]), 2
        )
        correct_prob_pairs = tf.concat(
            [correct_prob_tiled, tf.transpose(correct_prob_tiled, [1, 0, 2])], axis=2
        )
        incorrect_prob_pairs = tf.concat(
            [incorrect_prob_tiled, tf.transpose(incorrect_prob_tiled, [1, 0, 2])],
            axis=2,
        )
        correct_prob_tiled_1 = tf.expand_dims(
            tf.tile(tf.expand_dims(tensor1, 1), [1, tf.shape(tensor2)[0]]), 2
        )
        incorrect_prob_tiled_1 = tf.expand_dims(
            tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor1)[0]]), 2
        )
        correct_incorrect_pairs = tf.concat(
            [correct_prob_tiled_1, tf.transpose(incorrect_prob_tiled_1, [1, 0, 2])],
            axis=2,
        )
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs = get_pairs(
        correct_prob, incorrect_prob
    )
    correct_kernel = tf_kernel(correct_prob_pairs)
    incorrect_kernel = tf_kernel(incorrect_prob_pairs)
    correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs)
    sampling_weights_correct = tf.matmul(
        tf.expand_dims(1.0 - correct_prob, 1),
        tf.transpose(tf.expand_dims(1.0 - correct_prob, 1)),
    )
    correct_correct_vals = tf_get_out_tensor(correct_kernel, sampling_weights_correct)
    sampling_weights_incorrect = tf.matmul(
        tf.expand_dims(incorrect_prob, 1),
        tf.transpose(tf.expand_dims(incorrect_prob, 1)),
    )
    incorrect_incorrect_vals = tf_get_out_tensor(
        incorrect_kernel, sampling_weights_incorrect
    )
    sampling_correct_incorrect = tf.matmul(
        tf.expand_dims(1.0 - correct_prob, 1),
        tf.transpose(tf.expand_dims(incorrect_prob, 1)),
    )
    correct_incorrect_vals = tf_get_out_tensor(
        correct_incorrect_kernel, sampling_correct_incorrect
    )
    correct_denom = tf.reduce_sum(1.0 - correct_prob)
    incorrect_denom = tf.reduce_sum(incorrect_prob)
    m = tf.reduce_sum(correct_mask)
    n = tf.reduce_sum(1.0 - correct_mask)
    mmd_error = 1.0 / (m * m + 1e-5) * tf.reduce_sum(correct_correct_vals)
    mmd_error += 1.0 / (n * n + 1e-5) * tf.reduce_sum(incorrect_incorrect_vals)
    mmd_error -= 2.0 / (m * n + 1e-5) * tf.reduce_sum(correct_incorrect_vals)
    return tf.maximum(
        tf.stop_gradient(tf.cast(cond_k * cond_k_p, tf.float32)) * tf.sqrt(mmd_error + 1e-10),
        0.0,
    )



@given(batch=st.integers(min_value=2, max_value=100), n_classes=st.integers(min_value=2, max_value=100))
def test_pytorch_mmce(batch: int, n_classes: int):
    logits = torch.log(torch.distributions.Dirichlet(torch.ones(n_classes)).sample((batch,)))
    labels = torch.randint(0, n_classes, (batch,))


    tf_mmce_loss = tf_calibration_mmce_w_loss(logits, labels)
    torch_mmce_loss = calibration_mmce_w_loss(logits, labels)

    # If tf MMCE is NaN we should expect torch MMCE to be 0
    if tf.math.is_nan(tf_mmce_loss):
        assert torch_mmce_loss.item() == 0.0
    else:
        assert torch.isclose(
            torch_mmce_loss, torch.tensor(tf_mmce_loss.numpy()), atol=1e-6
        )