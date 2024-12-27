from flemme.config import module_config
supported_graph_encoders = {}
if module_config['graph']:
    from .gnn import *
    supported_graph_encoders['GCN'] = (GCNEncoder, GraphDecoder)
    supported_graph_encoders['Cheb'] = (ChebEncoder, GraphDecoder)
    supported_graph_encoders['GTrans'] = (TransConvEncoder, GraphDecoder)
