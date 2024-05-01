# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from numba import jit

import cProfile
import pstats

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = "./data/"

@jit(nopython=True)
def calculate_numerical_features(amounts, times, current_time, time_spans):
    # Calculate numerical features efficiently using Numba
    results = []
    for length in time_spans:
        # Calculate time bounds
        low_bound = current_time - length
        mask = (times >= low_bound) & (times <= current_time)
        
        if mask.sum() > 0:
            relevant_amounts = amounts[mask]
            mean_amount = relevant_amounts.mean()
            total_amount = relevant_amounts.sum()
            std_amount = relevant_amounts.std()
            count = len(relevant_amounts)
        else:
            mean_amount = total_amount = std_amount = 0
            count = 0
        
        results.append((mean_amount, total_amount, std_amount, count))
    return results

def featmap_gen(tmp_df):
    # Define time spans
    time_spans = np.array([2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000])
    time_names = [str(i) for i in time_spans]
    
    # Prepare arrays for fast access
    times = tmp_df['Time'].values
    amounts = tmp_df['Amount'].values

    feature_list = []

    # Initialize DataFrame for results
    feature_data = pd.DataFrame(index=tmp_df.index)
    
    for i, row in tqdm(tmp_df.iterrows(), total=len(tmp_df)):
        current_time = row['Time']
        current_amount = row['Amount']

        feature_dict = {}
        
        # Calculate numerical features
        numerical_results = calculate_numerical_features(amounts, times, current_time, time_spans)
        
        # Populate the DataFrame
        for (mean_amount, total_amount, std_amount, count), tname in zip(numerical_results, time_names):
            feature_dict[f'trans_at_avg_{tname}'] = mean_amount

            feature_dict[f'trans_at_totl_{tname}'] = total_amount

            feature_dict[f'trans_at_std_{tname}'] = std_amount

            feature_dict[f'trans_at_bias_{tname}'] = current_amount - mean_amount

            feature_dict[f'trans_at_num_{tname}'] = count
            
            # Calculate unique features using pandas
            valid_indices = np.where((times >= current_time - int(tname)) & (times <= current_time))[0]
            valid_data = tmp_df.iloc[valid_indices]
            feature_dict[f'trans_target_num_{tname}'] = valid_data['Target'].nunique()

            feature_dict[f'trans_location_num_{tname}'] = valid_data['Location'].nunique()

            feature_dict[f'trans_type_num_{tname}'] = valid_data['Type'].nunique()
    
        feature_list.append(feature_dict)



    # Create DataFrame from list of dictionaries

    feature_data = pd.DataFrame(feature_list, index=tmp_df.index)


    new_tmp_df = pd.concat([tmp_df, feature_data], axis=1)

    return new_tmp_df


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    # return mind + (data - mind) / (maxd - mind)
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map(graph,edge_feat):
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            # edge_feat[neighs_1_of_center, 0].std().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            # edge_feat[neighs_2_of_center, 0].std().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            # edge_feat[neighs_1_of_center, 1].std().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
            # edge_feat[neighs_2_of_center, 1].std().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


def main():

    set_seed(42)

    # %%
    """
        For Yelpchi dataset
        Code partially from https://github.com/YingtongDou/CARE-GNN
    """
    print(f"processing YELP data...")
    yelp = loadmat(os.path.join(DATADIR, 'YelpChi.mat'))
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']

    sparse_to_adjlist(net_rur, os.path.join(
        DATADIR, "yelp_rur_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rtr_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rsr_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_homo_adjlists.pickle"))

    data_file = yelp
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    # load the preprocessed adj_lists
    with open(os.path.join(DATADIR, "yelp_homo_adjlists.pickle"), 'rb') as file:
        homo = pickle.load(file)
    file.close()
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-yelp.bin", [g])

    # %%
    """
        For Amazon dataset
    """
    print(f"processing AMAZON data...")
    amz = loadmat(os.path.join(DATADIR, 'Amazon.mat'))
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    amz_homo = amz['homo']

    sparse_to_adjlist(net_upu, os.path.join(
        DATADIR, "amz_upu_adjlists.pickle"))
    sparse_to_adjlist(net_usu, os.path.join(
        DATADIR, "amz_usu_adjlists.pickle"))
    sparse_to_adjlist(net_uvu, os.path.join(
        DATADIR, "amz_uvu_adjlists.pickle"))
    sparse_to_adjlist(amz_homo, os.path.join(
        DATADIR, "amz_homo_adjlists.pickle"))

    data_file = amz
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    # load the preprocessed adj_lists
    with open(DATADIR + 'amz_homo_adjlists.pickle', 'rb') as file:
        homo = pickle.load(file)
    file.close()
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-amazon.bin", [g])

    # # %%
    # """
    #     For S-FFSD dataset
    # """
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    g.ndata['label'] = torch.from_numpy(
        labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # generate neighbor riskstat features
    for file_name in ['S-FFSD', 'yelp', 'amazon']:
        print(
            f"Generating neighbor risk-aware features for {file_name} dataset...")
        graph = dgl.load_graphs(DATADIR + "graph-" + file_name + ".bin")[0][0]
        graph: dgl.DGLGraph
        print(f"graph info: {graph}")

        edge_feat: torch.Tensor
        degree_feat = graph.in_degrees().unsqueeze_(1).float()
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

        origin_feat_name = []
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
        origin_feat_name = ['degree', 'riskstat']

        features_neigh, feat_names = feat_map(graph,edge_feat)
        # print(f"feature neigh: {features_neigh.shape}")

        features_neigh = torch.cat(
            (edge_feat, features_neigh), dim=1
        ).numpy()
        feat_names = origin_feat_name + feat_names
        features_neigh[np.isnan(features_neigh)] = 0.

        output_path = DATADIR + file_name + "_neigh_feat.csv"
        features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
        scaler = StandardScaler()
        # features_neigh = np.log(features_neigh + 1)
        features_neigh = pd.DataFrame(scaler.fit_transform(
            features_neigh), columns=features_neigh.columns)

        features_neigh.to_csv(output_path, index=False)

  

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Specify the filename to save the profiling results
    profile_filename = "profile_results.prof"
    
    # Dump the profiling statistics to the specified file
    profiler.dump_stats(profile_filename)
    
    # Create a Stats object from the profiling data
    stats = pstats.Stats(profile_filename)
    
    # Sort the statistics by cumulative time
    stats.sort_stats('cumtime')
    
    # Open a file to save the profiling report
    with open("profile_report.txt", "w") as report_file:
        # Redirect the standard output to the file
        stats.stream = report_file
        # Print the profiling report to the file
        stats.print_stats()
