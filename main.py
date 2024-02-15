from tracegnn.data.trace_graph import load_trace_csv, df_to_trace_graphs, TraceGraphIDManager, TraceGraph
from tracegnn.data.graph_to_vector import graph_to_dgl
import dgl
from typing import *
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.trainer import trainer
import torch
import pandas as pd
import numpy as np
import json
import yaml

def reset_exp_config():
    # Reset ExpConfig attributes
    ExpConfig.device = 'cpu'
    ExpConfig.dataset = 'dataset_b'
    ExpConfig.test_dataset = 'test'
    ExpConfig.seed = 1234
    ExpConfig.batch_size = 128
    ExpConfig.test_batch_size = 64
    ExpConfig.max_epochs = 80
    ExpConfig.enable_tqdm = True
    ExpConfig.dataset_root_dir = 'dataset'
    ExpConfig.decoder_max_nodes = 100

    # Reset Latency class attributes
    ExpConfig.Latency.embedding_type = 'normal'
    ExpConfig.Latency.latency_feature_length = 1
    ExpConfig.Latency.latency_embedding = 10
    ExpConfig.Latency.latency_max_value = 50.0

    # Reset Model class attributes
    ExpConfig.Model.vae = True
    ExpConfig.Model.anneal = False
    ExpConfig.Model.kl_weight = 1e-2
    ExpConfig.Model.n_z = 5
    ExpConfig.Model.latency_model = 'tree'
    ExpConfig.Model.structure_model = 'tree'
    ExpConfig.Model.latency_input = False
    ExpConfig.Model.embedding_size = 4
    ExpConfig.Model.graph_embedding_size = 4
    ExpConfig.Model.decoder_feature_size = 4
    ExpConfig.Model.latency_feature_size = 4
    ExpConfig.Model.latency_gcn_layers = 5

    # Reset RuntimeInfo and DatasetParams class attributes
    ExpConfig.RuntimeInfo.latency_range = None
    ExpConfig.RuntimeInfo.latency_p98 = None
    ExpConfig.DatasetParams.operation_cnt = None
    ExpConfig.DatasetParams.service_cnt = None
    ExpConfig.DatasetParams.status_cnt = None

def init_config(id_manager, train_graphs):
    # Set DatasetParams
    ExpConfig.DatasetParams.operation_cnt = id_manager.num_operations
    ExpConfig.DatasetParams.service_cnt = id_manager.num_services
    ExpConfig.DatasetParams.status_cnt = id_manager.num_status

    # Set runtime info
    if ExpConfig.RuntimeInfo.latency_range is None:
        tmp_latency_dict: Dict[int,List[float]] = {}

        ExpConfig.RuntimeInfo.latency_range = torch.zeros([ExpConfig.DatasetParams.operation_cnt + 1, 2], dtype=torch.float)
        ExpConfig.RuntimeInfo.latency_p98 = torch.zeros([ExpConfig.DatasetParams.operation_cnt + 1], dtype=torch.float)
        
        # TODO: Set default value
        ExpConfig.RuntimeInfo.latency_range[:, :] = 50.0

        for g in train_graphs:
            for _, nd in g.iter_bfs():
                tmp_latency_dict.setdefault(nd.operation_id, [])
                tmp_latency_dict[nd.operation_id].append(nd.features.avg_latency)
        for op, vals in tmp_latency_dict.items():
            vals_p99 = np.percentile(vals, 99)
            vals = np.array(vals)
            if np.any(vals < vals_p99):
                vals = vals[vals < vals_p99]

            # Set a minimum value for vals to avoid nan
            
            # TODO: set this
            vals_mean, vals_std = np.mean(vals), max(np.std(vals), 10.0)
            # vals_mean, vals_std = np.mean(vals), np.std(vals)

            ExpConfig.RuntimeInfo.latency_range[op] = torch.tensor([vals_mean, vals_std])
            ExpConfig.RuntimeInfo.latency_p98[op] = np.percentile(vals, 98)
        
        ExpConfig.RuntimeInfo.latency_range = ExpConfig.RuntimeInfo.latency_range.to(ExpConfig.device)
        ExpConfig.RuntimeInfo.latency_p98 = ExpConfig.RuntimeInfo.latency_p98.to(ExpConfig.device)

def split_trace_id(trace_id: str) -> Tuple[str, str]:
    # Assuming the trace_id format and deciding to split by the last 16 characters for low and the rest for high
    # This is a simplified example.
    trace_id_stripped = trace_id.replace("-", "")  # Remove dashes if any
    trace_id_high = trace_id_stripped[:-16]  # Get the first part of the trace_id
    trace_id_low = trace_id_stripped[-16:]  # Get the last 16 characters for the low part
    return trace_id_high, trace_id_low

def append_segment_data(segment, parent_name='', trace_id_high='', trace_id_low='', parent_span_id="0", gtrace_data=[]):
    # Check if segment is already a dictionary (i.e., a subsegment) or if it's a string that needs JSON deserialization
    if 'Document' not in segment and isinstance(segment, dict):
        doc = segment
    else:
        doc = json.loads(segment['Document'])
    if doc.get('in_progress'):
        return
    
    span_id = doc['id']
    service_name = doc['name']
    operation_name = f"{parent_name}/{service_name}" if parent_name else service_name
    start_time = doc['start_time']
    end_time = doc['end_time']
    duration = (end_time - start_time) * 1000
    nanosecond = int((duration - int(duration)) * 1e6)
    db_hash = 0  # Placeholder
    status = doc.get('http', {}).get('response', {}).get('status', 0)

    # Append current segment or subsegment data
    gtrace_data.append([trace_id_high, trace_id_low, span_id, parent_span_id, service_name, operation_name.strip('/'),
                        start_time, duration, nanosecond, db_hash, status])

    # Recurse for subsegments
    for subsegment in doc.get('subsegments', []):
        append_segment_data(subsegment, operation_name, trace_id_high, trace_id_low, span_id, gtrace_data)

def convert_to_gtrace(trace_logs: dict):
    gtrace_data = []
    for trace_id, logs in trace_logs.items():
        logs = logs[1]  # Assuming the first index holds the timestamp
        trace_id_high, trace_id_low = split_trace_id(trace_id)

        for segment in logs['Segments']:
            append_segment_data(segment, '', trace_id_high, trace_id_low, "0", gtrace_data)

    columns = ['traceIdHigh', 'traceIdLow', 'spanId', 'parentSpanId', 'serviceName', 'operationName', 'startTime',
               'duration', 'nanosecond', 'DBhash', 'status']
    return pd.DataFrame(gtrace_data, columns=columns)

def load_data():
    with open('trace_logs.json') as f:
        trace_logs = json.load(f)

    gtrace_data = convert_to_gtrace(trace_logs)

    # Convert Unix timestamp to datetime
    gtrace_data['startTime'] = pd.to_datetime(gtrace_data['startTime'], unit='s')
    # Format datetime into the specified string format
    gtrace_data['startTime'] = gtrace_data['startTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return gtrace_data

class Dataset(dgl.data.DGLDataset):
    def __init__(self, trace_graphs: List[TraceGraph]):
        self.trace_graphs = trace_graphs

    def process(self):
        pass

    def __len__(self):
        return len(self.trace_graphs)

    def __getitem__(self, idx):
        return graph_to_dgl(self.trace_graphs[idx])

def save_yml(field_values, filename):
    field_id = {'': 0}
    for i, value in enumerate(field_values):
        field_id[value] = i+1

    yaml_data = yaml.dump(field_id, default_flow_style=False)
    with open(f'{filename}.yml', 'w') as f:
        f.write(yaml_data)

def store_ids(gtrace_data):
    operations = gtrace_data.operationName.astype(str).unique()
    services = gtrace_data.serviceName.astype(str).unique()
    statuses = gtrace_data.status.astype(str).unique()

    save_yml(field_values=operations, filename='operation_id')
    save_yml(field_values=services, filename='service_id')
    save_yml(field_values=statuses, filename='status_id')

def main():
    gtrace_data = load_data()

    # Create yml files
    store_ids(gtrace_data)

    id_manager = TraceGraphIDManager(root_dir='.')
    trace_graphs: List[TraceGraph] = df_to_trace_graphs(gtrace_data, id_manager)

    # Split trace_graphs into train, val, and test groups
    total = len(trace_graphs)
    train_graphs = trace_graphs[: int(total*0.8)]
    val_graphs = trace_graphs[int(total*0.8):]

    init_config(id_manager, train_graphs)

    train_dataset = Dataset(train_graphs)
    val_dataset = Dataset(val_graphs)

    train_loader = dgl.dataloading.GraphDataLoader(
        train_dataset, batch_size=ExpConfig.batch_size, shuffle=True)

    val_loader = dgl.dataloading.GraphDataLoader(
        val_dataset, batch_size=ExpConfig.batch_size, shuffle=True)

    test_loader = None
    
    trainer(ExpConfig, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()