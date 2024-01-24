import argparse as ap
import logging
import os
import sys
from sklearn import metrics

import atom3d.datasets as da
import dotenv as de
import pandas as pd
import pathlib
import pytorch_lightning as pl
import torch_geometric

import ares.data as d
import ares.model as m
#import ares.model_2 as m

root_dir = pathlib.Path(__file__).parent.absolute()
de.load_dotenv(os.path.join(root_dir, '.env'))
logger = logging.getLogger("lightning")

def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results

def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('dataset', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--nolabels', dest='use_labels', action='store_false')
    parser.add_argument('--num_workers', type=int, default=1)

    # add model specific args
    parser = m.ARESModel.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    transform = d.create_transform(
        hparams.use_labels, hparams.label_dir, hparams.filetype)

    # DATA PREP
    logger.info(f"Dataset of type {hparams.filetype}")
    dataset = da.load_dataset(hparams.dataset, hparams.filetype, transform)
    dataloader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    # MODEL SETUP
    logger.info("Loading model weights...")
    tfnn = m.ARESModel.load_from_checkpoint(hparams.checkpoint_path)

    trainer = pl.Trainer.from_argparse_args(hparams)

    # PREDICTION
    logger.info("Running prediction...")
    trainer.test(tfnn, dataloader, verbose=False)




    # SAVE OUTPUT
    pd.DataFrame(tfnn.predictions).to_csv(
        hparams.output_file, index=False, float_format='%.7f')




if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()
