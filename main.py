import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)
# sys.path.append("..")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    yaml_file = "config/gtan_cfg.yaml"
    

    # config = Config().get_config()
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    return args


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # for ICONIP16 & AAAI20
    if args['method'] == 'stan':
        if os.path.exists("data/tel_3d.npy"):
            return
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return
        features, labels = span_data_2d(feat_df)
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):
   
    if args['method'] == 'gtan':
        from methods.gtan.gtan_main import gtan_main, load_gtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            args['dataset'], args['test_size'])
        gtan_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)
    elif args['method'] == 'gtan_opt':
        from methods.gtan.gtan_main_og import gtan_main_og, load_gtan_data_og
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data_og(
            args['dataset'], args['test_size'])
        gtan_main_og(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)
    else:
        raise NotImplementedError("Unsupported method. ")


if __name__ == "__main__":
    main(parse_args())
