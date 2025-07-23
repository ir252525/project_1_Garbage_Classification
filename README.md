# project_1_Garbage_Classification
 A Vision Classification model for Classifing Garbages 


by using efficientnet_b1 pretrained model I achieved 95 % accuracy in 10 epoches

hyperparameters are as followed:

1 batch size = 32

2 Freezing efficientnet b1 parameters except block 8

3 epochs = 10

4 learning rate for Adam optimizer = 0.001

5 weight_decay = 1e-5
________________________________________________________________________________________

train total time on cuda: 11 Minutes

'test_loss': 0.1368
'test_acc': 0.95

time_per_pred_cpu: 0.0816
