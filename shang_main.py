import utils
from tqdm import tqdm
import copy
import dataset.DataLoader as DataLoader
import train
from train import train_model,validate_model,build_model_and_optimizer
import torch
import logging
import matplotlib.pyplot as plt

if __name__=="__main__":
    logger = utils.Logger(log_dir='logs', log_file='my_log.log', log_level=logging.INFO)
    device="cuda" if torch.cuda.is_available() else "cpu"

    ## task 1 
    target_channels=["RGB","RG","GB","R","G","B"]
    batch_size=128
    num_workers=16
    num_classes=50
    num_epochs=10

    logger.info(f'Batch size: {batch_size}')
    logger.info(f'num workers: {num_workers}')
    # ## train baseline baseline_model
    # train.baseline_model_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels)
    # ## my cool model
    # train.my_cool_model_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels)
    ## channel processing method
   # train.channel_processing_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels)



    ## task 2
    train.task2_resnet34(logger,batch_size,num_workers,num_classes,device,num_epochs)
    # train.task2_cool_model(logger,batch_size,num_workers,num_classes,device,num_epochs)
    

