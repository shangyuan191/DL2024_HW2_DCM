import logging
import os
import matplotlib.pyplot as plt
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

def plot_result(epochs,train_losss,val_losss,val_accuracys,channel_method,target_channel):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losss, 'b', label='Training loss')
        plt.plot(epochs, val_losss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracys, 'b', label='Validation accuracy')
        plt.title('Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plot/{target_channel}_{channel_method}_result.png')
        plt.close()

