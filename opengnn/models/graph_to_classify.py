from typing import Any, Dict, Tuple

import tensorflow as tf

import opengnn.constants as constants
from opengnn.inputters.token_embedder import TokenEmbedder
from opengnn.inputters.keywords_extractor_inputter import KeywordsExtractorInputter
from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.models.model import Model
from opengnn.utils.metrics import bleu_score, rouge_2_fscore, f1_score
from opengnn.utils.ops import batch_gather


class GraphToClassify(Model):
    def __init__(self,
                 source_inputter: KeywordsExtractorInputter,
                 target_inputter: TokenEmbedder,
                 encoder: GraphEncoder,
                 name: str,
                 metrics: Tuple[str] = ('BLEU', 'ROUGE'),
                 only_attend_primary: bool = True):
        super().__init__(name, source_inputter, target_inputter)
        self.encoder = encoder
        self.metrics = metrics
        self.only_attend_primary = only_attend_primary

    def __call__(self,
                 features: Dict[str, tf.Tensor],
                 labels: Dict[str, tf.Tensor],
                 mode: tf.estimator.ModeKeys,
                 params: Dict[str, Any],
                 config=None,
                 session=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            features: [description]
            labels: [description]
            mode: [description]
            params: [description]
            config: [description]
        Returns:
            outputs: raw outputs
            predictions: predictions
        """
        adj_matrices = features["graph"]
        node_features = features["features"]
        graph_sizes = features["length"]

        primary_paths = features["primary_path"]
        primary_path_lengths = features["primary_path_length"]
        target_vocab_size = self.labels_inputter.vocabulary_size

        # format input features (ex: embedding labels)
        node_features = self.features_inputter.transform(
            (node_features, graph_sizes), mode)

        batch_size = tf.shape(node_features, out_type=tf.int64)[0]
        max_num_nodes = tf.shape(node_features, out_type=tf.int64)[1]

        # build encoder using inputter metadata manually
        # this is due to https://github.com/tensorflow/tensorflow/issues/15624
        # and to the way estimators need to rebuild variables

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.encoder.build(
                self.features_inputter.node_features_size,
                self.features_inputter.num_edge_types,
                mode=mode)

            representations, initial_state = self.encoder(
                adj_matrices=adj_matrices,
                node_features=node_features,
                graph_sizes=graph_sizes,
                mode=mode)

        with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
            classify_linear = tf.layers.Dense(1, use_bias=False)
            classify_linear.build((None, 512))
        expanded_state = tf.tile(
            tf.reshape(initial_state, [batch_size, 1, -1]), [1, max_num_nodes, 1])
        logits = classify_linear(
            tf.concat([representations, expanded_state], axis=-1)
        )
        logits = tf.squeeze(logits)

        str_tokens = features['str_tokens']

        # Fetch tokens from normal vocab
        rev_vocab = tf.contrib.lookup.index_to_string_table_from_file(
            self.labels_inputter.vocabulary_file,
            vocab_size=self.labels_inputter.vocabulary_size - 1,
            default_value=constants.UNKNOWN_TOKEN)

        zeros = tf.zeros_like(str_tokens, dtype=tf.int64)
        zero_tokens = rev_vocab.lookup(zeros)

        target_tokens = tf.where(
            tf.greater_equal(logits, 0.5),
            str_tokens,
            zero_tokens)

        predictions = {
            "tokens": target_tokens,
        }

        return logits, predictions

    def compute_loss(self, features, labels, outputs, params, mode: tf.estimator.ModeKeys) -> tf.Tensor:
        # extract labels and batch info

        label_ids = features["mul_task_id"]
        sequence_lens = features["mul_task_id_length"]

        batch_size = tf.shape(outputs)[0]
        batch_max_len = tf.shape(label_ids)[1]

        loss_per_time = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_ids, tf.float32),
            logits=outputs)

        weights = tf.sequence_mask(
            sequence_lens, maxlen=batch_max_len, dtype=tf.float32)

        unnorm_loss = tf.reduce_sum(loss_per_time * weights)
        total_timesteps = tf.reduce_sum(tf.cast(sequence_lens, tf.float32))

        if params.get("average_loss_in_time", False):
            loss = unnorm_loss / total_timesteps
            tb_loss = loss
        else:
            loss = unnorm_loss / tf.cast(batch_size, tf.float32)
            tb_loss = unnorm_loss / total_timesteps

        return loss, tb_loss

    def compute_metrics(self, _, labels, predictions):
        # extract labels and batch info
        labels_ids = labels["ids_out"]
        predictions_ids = predictions['ids']

        eval_metric_ops = {}

        if "BLEU" in self.metrics:
            eval_metric_ops["bleu"] = bleu_score(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        if "ROUGE" in self.metrics:
            eval_metric_ops["rouge"] = rouge_2_fscore(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        if "F1" in self.metrics:
            eval_metric_ops["f1"] = f1_score(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        return eval_metric_ops

    def process_prediction(self, prediction):
        prediction_tokens = [token.decode('utf-8') for token in prediction['tokens']]
        prediction_tokens = list(set(prediction_tokens))
        stopwords = ['<blank>', '<s>']
        for token in prediction_tokens[:]:
            if len(token.strip()) == 0 or token in stopwords:
                prediction_tokens.remove(token)
        return prediction_tokens
