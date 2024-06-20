import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from tqdm import tqdm
from my_model import my_cool_model
import dataset.DataLoader as DataLoader
import utils


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    correct_predictions=0
    total_predictions=0
    for images, labels in tqdm(train_loader,desc='train_loader'):
        images, labels = images.to(device).float(), labels.to(device)  # Ensure images are FloatTensor
        optimizer.zero_grad()
        with torch.autocast(device_type=device,enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss/100,epoch_accuracy
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    val_loss=0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader,desc='validate_loader'):
            images, labels = images.to(device).float(), labels.to(device)  # Ensure images are FloatTensor
            outputs = model(images)
            loss=criterion(outputs,labels)
            val_loss+=loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss/=total
    val_accuracy = correct / total
    return val_loss/100,val_accuracy
def build_model_and_optimizer(target_channel, num_classes, device, model_type):
    if model_type == "resnet34":
        if target_channel == "RGB":
            model = resnet34(weights=False)  # Use updated argument for weights
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet34(weights='IMAGENET1K_V1')
            model.conv1 = nn.Conv2d(len(target_channel), 64, kernel_size=3, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type=='my_cool_model':
        model=my_cool_model.DynamicInputCNN(target_channel,num_classes)
    elif model_type=='baseline_model':
        model=my_cool_model.SimpleCNN(len(target_channel),num_classes)
    elif model_type=="channel_processing":
        model=my_cool_model.SimpleCNN(len(target_channel),num_classes)
    elif model_type=="task2_cool_model":
        model=my_cool_model.Task2CoolModel(len(target_channel),num_classes)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    return model, criterion, optimizer


def baseline_model_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels):
    ## train baseline baseline_model
    logger.info(f"Train baseline model")
    target_channel="RGB"
    channel_method="baseline_model"
    best_model_path= f"./my_model/best_model_{channel_method}_{target_channel}_NumEpoch_{num_epochs}.pth"
    train_loader=DataLoader.build_train_loader(batch_size,num_workers,target_channel,channel_method)
    val_loader=DataLoader.build_val_loader(batch_size,num_workers,target_channel,channel_method)
    model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="baseline_model")
    train_losss=[]
    train_accuracys=[]
    val_losss=[]
    val_accuracys=[]
    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")
        train_loss,train_accuracy=train_model(model,train_loader,criterion,optimizer,device)
        logger.info(f"Training loss: {train_loss}")
        train_losss.append(train_loss)
        train_accuracys.append(train_accuracy)
        val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
        logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        val_losss.append(val_loss)
        val_accuracys.append(val_accuracy)
        # Save the best model
        best_val_loss = utils.save_model(model, epoch+1, val_loss,val_accuracy, best_val_loss, channel_method, target_channel,best_model_path)

    utils.plot_result(range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    utils.write_train_result(f'./result/train_result/{channel_method}_{target_channel}_NumEpoch_{num_epochs}.csv',range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    # Load the best model
    # 加载保存的模型字典
    checkpoint = torch.load(best_model_path)
    # 初始化模型
    model, _, _ = build_model_and_optimizer(target_channel, num_classes, device, model_type="baseline_model")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    model.eval()
    test_losss=[]
    test_accuracys=[]
    for target_channel in tqdm(target_channels, total=len(target_channels)):
        test_loader = DataLoader.build_test_loader(batch_size, num_workers, target_channel,channel_method)
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        logger.info(f"Target channel: {target_channel}\nTest loss: {test_loss}, Test accuracy: {test_accuracy}")
        test_losss.append(test_loss)
        test_accuracys.append(test_accuracy)
    utils.write_test_result(f'./result/test_result/{channel_method}_NumEpoch_{num_epochs}.csv',target_channels,test_losss,test_accuracys,channel_method)

def my_cool_model_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels):
    logger.info(f"Train my cool model")
    target_channel="RGB"
    channel_method="my_cool_model"
    best_model_path= f"./my_model/best_model_{channel_method}_{target_channel}_NumEpoch_{num_epochs}.pth"
    train_loader=DataLoader.build_train_loader(batch_size,num_workers,target_channel,channel_method)
    val_loader=DataLoader.build_val_loader(batch_size,num_workers,target_channel,channel_method)
    model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="my_cool_model")
    train_losss=[]
    train_accuracys=[]
    val_losss=[]
    val_accuracys=[]
    best_val_loss = float('inf')  # Initialize best validation loss
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")
        train_loss,train_accuracy=train_model(model,train_loader,criterion,optimizer,device)
        logger.info(f"Training loss: {train_loss}")
        train_losss.append(train_loss)
        train_accuracys.append(train_accuracy)
        val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
        logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        val_losss.append(val_loss)
        val_accuracys.append(val_accuracy)
        # Save the best model
        best_val_loss = utils.save_model(model, epoch+1, val_loss,val_accuracy, best_val_loss, channel_method, target_channel,best_model_path)
    utils.plot_result(range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    utils.write_train_result(f'./result/train_result/{channel_method}_{target_channel}_NumEpoch_{num_epochs}.csv',range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    # Load the best model
    # 加载保存的模型字典
    checkpoint = torch.load(best_model_path)
    # 初始化模型
    
    model, _, _ = build_model_and_optimizer(target_channel, num_classes, device, model_type="my_cool_model")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    model.eval()
    test_losss=[]
    test_accuracys=[]
    for target_channel in tqdm(target_channels, total=len(target_channels)):
        test_loader = DataLoader.build_test_loader(batch_size, num_workers, target_channel,channel_method)
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        logger.info(f"Target channel: {target_channel}\nTest loss: {test_loss}, Test accuracy: {test_accuracy}")
        test_losss.append(test_loss)
        test_accuracys.append(test_accuracy)
    utils.write_test_result(f'./result/test_result/{channel_method}_NumEpoch_{num_epochs}.csv',target_channels,test_losss,test_accuracys,channel_method)


def channel_processing_method(logger,batch_size,num_workers,num_classes,device,num_epochs,target_channels):
    ## preprocessing method
    logger.info('Use data preprocessing method')
    for channel_method in ['DIY','sigmoid']: 
        logger.info(f'Channel method : {channel_method}')
        test_losss=[]
        test_accuracys=[]
        for target_channel in tqdm(target_channels,total=len(target_channels)):
            logger.info(f'target channel : {target_channel}')
            best_model_path= f"./my_model/best_model_{channel_method}_{target_channel}_NumEpoch_{num_epochs}.pth"
            train_loader=DataLoader.build_train_loader(batch_size,num_workers,target_channel,channel_method)
            val_loader=DataLoader.build_val_loader(batch_size,num_workers,target_channel,channel_method)
            test_loader=DataLoader.build_test_loader(batch_size,num_workers,target_channel,channel_method)
            model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="channel_processing")
            train_losss=[]
            train_accuracys=[]
            val_losss=[]
            val_accuracys=[]
            best_val_loss = float('inf')  # Initialize best validation loss
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")

                train_loss,train_accuracy=train_model(model,train_loader,criterion,optimizer,device)
                logger.info(f"Training loss: {train_loss}")
                train_losss.append(train_loss)
                train_accuracys.append(train_accuracy)
                val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
                logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
                val_losss.append(val_loss)
                val_accuracys.append(val_accuracy)
                # Save the best model
                best_val_loss = utils.save_model(model, epoch+1, val_loss,val_accuracy, best_val_loss, channel_method, target_channel,best_model_path)
            utils.plot_result(range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
            utils.write_train_result(f'./result/train_result/{channel_method}_{target_channel}_NumEpoch_{num_epochs}.csv',range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
            # Load the best model
            # 加载保存的模型字典
            checkpoint = torch.load(best_model_path)
            # 初始化模型
            
            model, _, _ = build_model_and_optimizer(target_channel, num_classes, device, model_type="baseline_model")
            model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
            
            model.eval()
            test_loss,test_accuracy=validate_model(model,test_loader,criterion,device)
            logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
            test_losss.append(test_loss)
            test_accuracys.append(test_accuracy)
        utils.write_test_result(f'./result/test_result/{channel_method}_NumEpoch_{num_epochs}.csv',target_channels,test_losss,test_accuracys,channel_method)

def task2_resnet34(logger,batch_size,num_workers,num_classes,device,num_epochs):
    ## train baseline resnet34 model
    logger.info('train baseline resnet34 model')
    target_channel="RGB"
    channel_method="task2_resnet34_model"
    best_model_path= f"./my_model/best_model_{channel_method}_{target_channel}_NumEpoch_{num_epochs}.pth"
    train_loader=DataLoader.build_train_loader(batch_size,num_workers,target_channel,channel_method)
    val_loader=DataLoader.build_val_loader(batch_size,num_workers,target_channel,channel_method)
    test_loader=DataLoader.build_test_loader(batch_size,num_workers,target_channel,channel_method)
    model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="resnet34")
    train_losss=[]
    train_accuracys=[]
    val_losss=[]
    val_accuracys=[]
    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")
        train_loss,train_accuracy=train_model(model,train_loader,criterion,optimizer,device)
        logger.info(f"Training loss: {train_loss}")
        train_losss.append(train_loss)
        train_accuracys.append(train_accuracy)
        val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
        logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        val_losss.append(val_loss)
        val_accuracys.append(val_accuracy)
        # Save the best model
        best_val_loss = utils.save_model(model, epoch+1, val_loss,val_accuracy, best_val_loss, channel_method, target_channel,best_model_path)

    utils.plot_result(range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    utils.write_train_result(f'./result/train_result/{channel_method}_{target_channel}_NumEpoch_{num_epochs}.csv',range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    # Load the best model
    # 加载保存的模型字典
    checkpoint = torch.load(best_model_path)
    # 初始化模型
    
    model, _, _ = build_model_and_optimizer(target_channel, num_classes, device, model_type="resnet34")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    model.eval()
    
    test_losss=[]
    test_accuracys=[]
    target_channels=['RGB']
    for target_channel in tqdm(target_channels, total=len(target_channels)):
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        logger.info(f"Target channel: {target_channel}\nTest loss: {test_loss}, Test accuracy: {test_accuracy}")
        test_losss.append(test_loss)
        test_accuracys.append(test_accuracy)
    utils.write_test_result(f'./result/test_result/{channel_method}_NumEpoch_{num_epochs}.csv',target_channels,test_losss,test_accuracys,channel_method)


def task2_cool_model(logger,batch_size,num_workers,num_classes,device,num_epochs):
    logger.info('train task2 cool model')
    target_channel="RGB"
    channel_method="task2_cool_model"
    best_model_path= f"./my_model/best_model_{channel_method}_{target_channel}_NumEpoch_{num_epochs}.pth"
    train_loader=DataLoader.build_train_loader(batch_size,num_workers,target_channel,channel_method)
    val_loader=DataLoader.build_val_loader(batch_size,num_workers,target_channel,channel_method)
    test_loader=DataLoader.build_test_loader(batch_size,num_workers,target_channel,channel_method)
    model,criterion,optimizer=build_model_and_optimizer(target_channel,num_classes,device,model_type="task2_cool_model")
    train_losss=[]
    train_accuracys=[]
    val_losss=[]
    val_accuracys=[]
    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} for target channel {target_channel}")
        train_loss,train_accuracy=train_model(model,train_loader,criterion,optimizer,device)
        logger.info(f"Training loss: {train_loss}")
        train_losss.append(train_loss)
        train_accuracys.append(train_accuracy)
        val_loss,val_accuracy=validate_model(model,val_loader,criterion,device)
        logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        val_losss.append(val_loss)
        val_accuracys.append(val_accuracy)
        # Save the best model
        best_val_loss = utils.save_model(model, epoch+1, val_loss,val_accuracy, best_val_loss, channel_method, target_channel,best_model_path)

    utils.plot_result(range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    utils.write_train_result(f'./result/train_result/{channel_method}_{target_channel}_NumEpoch_{num_epochs}.csv',range(1,num_epochs+1),train_losss,train_accuracys,val_losss,val_accuracys,channel_method,target_channel)
    # Load the best model
    # 加载保存的模型字典
    checkpoint = torch.load(best_model_path)
    # 初始化模型
    
    model, _, _ = build_model_and_optimizer(target_channel, num_classes, device, model_type="task2_cool_model")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    model.eval()
    
    test_losss=[]
    test_accuracys=[]
    target_channels=['RGB']
    for target_channel in tqdm(target_channels, total=len(target_channels)):
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        logger.info(f"Target channel: {target_channel}\nTest loss: {test_loss}, Test accuracy: {test_accuracy}")
        test_losss.append(test_loss)
        test_accuracys.append(test_accuracy)
    utils.write_test_result(f'./result/test_result/{channel_method}_NumEpoch_{num_epochs}.csv',target_channels,test_losss,test_accuracys,channel_method)
