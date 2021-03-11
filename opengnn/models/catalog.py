from opengnn.encoders.gcn_encoder import GCNEncoder
from opengnn.encoders.ggnn_encoder import GGNNEncoder
from opengnn.inputters.features_inputter import FeaturesInputter
from opengnn.inputters.graph_inputter import GraphEmbedder
from opengnn.inputters.token_embedder import TokenEmbedder
from opengnn.models.graph_regressor import GraphRegressor


class chemModelGGNN(GraphRegressor):
    def __init__(self):
        super().__init__(
            source_inputter=GraphEmbedder(
                edge_vocabulary_file_key="edge_vocabulary",
                node_embedder=TokenEmbedder(
                    vocabulary_file_key="node_vocabulary",
                    embedding_size=64)),
            target_inputter=FeaturesInputter(),
            encoder=GGNNEncoder(
                num_timesteps=[2, 2],
                node_feature_size=64),
            name="chemModelGGNN")


class chemModelGCN(GraphRegressor):
    def __init__(self):
        super().__init__(
            source_inputter=GraphEmbedder(
                edge_vocabulary_file_key="edge_vocabulary",
                node_embedder=TokenEmbedder(
                    vocabulary_file_key="node_vocabulary",
                    embedding_size=64)),
            target_inputter=FeaturesInputter(),
            encoder=GCNEncoder(
                layer_sizes=[64, 32, 16]),
            name="chemModelGCN")
