import argparse as ap
import logging
import os
import pathlib
import sys

import atom3d.datasets as da
import dotenv as de
import pytorch_lightning as pl
import pytorch_lightning.loggers as log
import torch_geometric
import wandb

import ares.data as d
import ares.model as m

from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import ModelCheckpoint


root_dir = pathlib.Path(__file__).parent.parent.absolute()
de.load_dotenv(os.path.join(root_dir, '.env'))
logger = logging.getLogger("lightning")

os.environ['SLURM_JOB_NAME'] = 'bash'


def main():
    #os.environ["WANDB_MODE"] = "offline"
    print("wandb init")
    wandb.init(project="ares_19DNAMD")
    print("wandb init finish")
    print("wandb log finish")
    
    #wandb.init(project="ares_19DNAMD", settings=wandb.Settings(start_method='thread', timeout=300))
    pl.seed_everything(1234)     ###   固定种子
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('train_dataset', type=str, default='/mnt/d/work_GNN/ARES/data/test/lmdbs/train')
    parser.add_argument('val_dataset', type=str, default='/mnt/d/work_GNN/ARES/data/test/lmdbs/val')
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    #parser.add_argument('--checkpoint_path',type=str, default="/home/zhangyi/3dRNA/ARES/ares_release/model_RNA/RNA3DCNN/ww6zavkk/checkpoints/bk/epoch=0.ckpt")
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default="/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/118DNA/d148-4_v2/ljdldjrg/checkpoints/epoch=73.ckpt")
    #parser.add_argument('--checkpoint_path', type=str,
    #                     default="/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/118DNA/d148-4_v2/6kfrqb8g/checkpoints/bk/epoch=66.ckpt")
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default="/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/118DNA/d148-4_v2/rzx1wfxq/checkpoints/epoch=18.ckpt")
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default="/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/118DNA/d148-4_v2/uivfbebv/checkpoints/epoch=27.ckpt")
    #parser.add_argument('--checkpoint_path', type=str,
    #                    default="/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/118DNA/d148-4_v2/n7rhg3hu/checkpoints/epoch=34.ckpt")

    # add model specific args
    parser = m.ARESModel.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    transform = d.create_transform(True, hparams.label_dir, hparams.filetype)

    # DATA PREP
    logger.info(f"Dataset of type {hparams.filetype}")

    logger.info(f"Creating dataloaders...")
    train_dataset = da.load_dataset(hparams.train_dataset, hparams.filetype,
                                    transform=transform)
    train_dataloader = torch_geometric.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)
    val_dataset = da.load_dataset(hparams.val_dataset, hparams.filetype,
                                  transform=transform)
    val_dataloader = torch_geometric.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)

    ######无预训练模型
    tfnn = m.ARESModel(**dict_args)
    ######加入预训练模型
    #tfnn = m.ARESModel.load_from_checkpoint(hparams.checkpoint_path)
    #print("pre_training")
    ########c
    os.environ['MODEL_DIR'] = '/home/zhangyi/3dRNA/ARES/ares_release/model_DNA/3dDNA'
    # os.environ['MODEL_DIR']='/mnt/d/work_GNN/ARES/model'

    wandb_logger = log.WandbLogger(save_dir=os.environ['MODEL_DIR'], name="1_0.01_10", project="id_128")
    #trainer = pl.Trainer.from_argparse_args(hparams, logger=wandb_logger, val_check_interval=1)


    # TRAINING
    logger.info("Running training...")
    print("################################")
    trainer = pl.Trainer.from_argparse_args(hparams,logger=wandb_logger)
    out = trainer.fit(tfnn, train_dataloader, val_dataloader)




if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()
