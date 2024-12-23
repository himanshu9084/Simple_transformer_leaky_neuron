_seed_ = 2020

 # nohup python custom_train.py -b 64 -epochs 1024 -model_name 'HybridSpiking_backbone' > default_train.log &

'''
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import re
import os 

def log_read(log_file):
    
    #24-11-2024 00:23:26 INFO     epoch = 0, train_loss = 2.2659, train_acc = 0.2526,
    #test_loss = 2.0863, test_acc = 0.3646, max_test_acc = 0.3646
    

    iters = []
    train_loss = []
    test_loss = []
    max_test = 0
    max_epoch = 0

    for line in log_file:
        if "epoch = " in line:
            split_line = line[29:].split(",")
            #print(split_line)
            #print(split_line.split(","))
            iter_val=re.findall(r'[\d\.\d]+', split_line[0])
            iters.append(int(iter_val[0]))
            train_loss_val=re.findall("\d+\.\d+", split_line[1])
            train_loss.append(float(train_loss_val[0]))
            test_loss_val=re.findall("\d+\.\d+", split_line[3])
            test_loss.append(float(test_loss_val[0]))
            
            max_val_curr = float(re.findall("\d+\.\d+", split_line[-1])[0])
            if max_val_curr>max_test:
                max_epoch = int(iter_val[0])
                max_test = max_val_curr

    print("Epochs completed : ",iters[-1])            
    print("Maximum test acc : ",max_test)
    print("Maximum test acc achieved at Epoch : ",max_epoch)
    plt.plot(iters, train_loss, label = "train_loss")
    plt.plot(iters, test_loss, label = "test_loss")
    plt.legend()
    plt.show()


log_path = "05122024_11_48_04"

exp_path = "/data01/ucidata/event_vision/spiking_NN/logs/"
file_name = exp_path+log_path+"/training.log"

with open(file_name) as log_file:
    log_file = log_file.readlines()
    
log_read(log_file)
'''