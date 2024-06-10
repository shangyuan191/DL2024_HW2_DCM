import utils
from tqdm import tqdm
import copy
import dataset.DataLoader as DataLoader
from train import train_model,validate_model,build_model_and_optimizer
import torch
import logging

if __name__=="__main__":
    logger = utils.Logger(log_dir='logs', log_file='my_log.log', log_level=logging.INFO)
    device="cuda" if torch.cuda.is_available() else "cpu"
    target_channels=["RG","GB","R","G","B"]
    batch_size=128
    num_workers=16
    num_classes=50
    num_epochs=30
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'num workers: {num_workers}')
    for target_channel in tqdm(target_channels,total=len(target_channels)):
        ### preprocessing method
        logger.info(f'target channel : {target_channel}')
        logger.info('Use data preprocessing method')
        for channel_method in ['DIY','sigmoid']:

            logger.info(f'Channel method : {channel_method}')
            train_loader,val_loader,test_loader=DataLoader.build_dataloader(batch_size=batch_size,num_workers=num_workers,target_channel=target_channel,channel_method=channel_method)

            model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="resnet34")
            train_losss=[]
            val_losss=[]
            val_accuracys=[]
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")

                train_loss=train_model(model,train_loader,criterion,optimizer,device)
                logger.info(f"Training loss: {train_loss}")
                train_losss.append(train_loss)
                val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
                logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
                val_losss.append(val_loss)
                val_accuracys.append(val_accuracy)
            test_loss,test_accuracy=validate_model(model,test_loader,criterion,device)
            logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
            utils.plot_result(range(1,num_epochs+1),train_losss,val_losss,val_accuracys,channel_method,target_channel)


        ### my cool model
        channel_method="my_cool_model"
        logger.info(f'Use {channel_method}')
        train_loader,val_loader,test_loader=DataLoader.build_dataloader(batch_size=64,num_workers=8,target_channel=target_channel,channel_method=channel_method)
        model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="my_cool_model")
        train_losss=[]
        val_losss=[]
        val_accuracys=[]
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")

            train_loss=train_model(model,train_loader,criterion,optimizer,device)
            logger.info(f"Training loss: {train_loss}")
            train_losss.append(train_loss)
            val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
            logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
            val_losss.append(val_loss)
            val_accuracys.append(val_accuracy)
        test_loss,test_accuracy=validate_model(model,test_loader,criterion,device)
        logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
        utils.plot_result(range(1,num_epochs+1),train_losss,val_losss,val_accuracys,channel_method,target_channel)

            




