import logging
import os
import matplotlib.pyplot as plt
import torch
import copy
import logging
import pandas as pd
class Logger:
    def __init__(self,log_dir='logs',log_file='my_log.log',log_level=logging.INFO):
        self.logger = logging.getLogger('MyLogger')
        self.logger.setLevel(log_level)

        # Create log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)

        # File handler to write logs to a file
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(log_level)

        # Console handler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter to specify the format of the logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Adding handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

def plot_result(epochs,train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losss, 'b', label='Training loss')
        plt.plot(epochs, val_losss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracys, 'b', label='train accuracy')
        plt.plot(epochs, val_accuracys, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plot/{target_channel}_{channel_method}_NumEpoch_{len(epochs)}_result.png')
        plt.close()

def save_model(model,epoch,val_loss,val_accuracy,best_val_loss,channel_method,target_channel,model_path):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'channel_method': channel_method,
            'target_channel': target_channel
        }, model_path)
    return best_val_loss

def write_train_result(file_name,epochs,train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel):
    data={'epochs':epochs,
         'train_loss':train_losss,
          'train_accuracy':train_accuracys,
          'validation_loss':val_losss,
          'validation_accuracy':val_accuracys}
    df=pd.DataFrame(data)
    df.to_csv(file_name,index=False)
    
def write_test_result(file_name,target_channels,test_losss,test_accuracys,channel_method):
    data={'target_channel':target_channels,
            'test_loss':test_losss,
            'test_accuracy':test_accuracys}
    df=pd.DataFrame(data)
    df.to_csv(file_name,index=False)     

