from torch.utils.data import Dataset
import scipy.io as sio
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
from model_ac_strain import myNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import h5py

# seed = 2
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)  # Numpy module.
# # random.seed(seed)  # Python random module.
# torch.manual_seed(seed)

                 #需要读取的mat文件路径


writer=SummaryWriter('logs')



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_root=r'D:\phycnn_ac\train'
# dataname=os.path.join(data_root,'tr_dataset2.mat')  #dataset_min 1650*21*800
dataname=os.path.join(data_root,'tr_dataset_u1_noise_change_2.mat')


class mydata(Dataset):
    def __init__(self,Data_input,Data_output):
        self.Data_input=Data_input
        self.Data_output=Data_output
    def __len__(self):
        return len(self.Data_input)
    def __getitem__(self, index):
        input=torch.FloatTensor(self.Data_input[index])
        output=torch.FloatTensor(self.Data_output[index])
        input = np.expand_dims(input, 0)
        output=np.expand_dims(output,0)
        return input,output



def dataset_sp(dataset,train_scale=0.8,valid_scale=0.2,test_scale=0):  #划分数据集
    #dataset 为三维数组，c*h*w
    data_num=dataset.shape[0]
    train_num=int(train_scale*data_num)
    valid_num = int(valid_scale * data_num)
    test_num = int(test_scale * data_num)
    np.random.shuffle(dataset)
    train_dataset=dataset[0:train_num,:,:]
    valid_dataset=dataset[train_num:train_num+valid_num,:,:]
    test_dataset=dataset[train_num+valid_num:train_num+valid_num+test_num,:,:]
    return train_dataset,valid_dataset,test_dataset

Data_a=h5py.File(dataname)               #读取mat文件
# Data = Data_a['tr_dataset1'][:]
Data = Data_a['tr_dataset1'][:]
all_data=np.transpose(Data,(0,2,1)) # 维度变换

train_dataset,valid_dataset,test_dataset=dataset_sp(all_data)
# #
# train_dataset=Data_a['train_dataset'][:]
# train_dataset=np.transpose(train_dataset,(0,2,1))
# valid_dataset=Data_a['valid_dataset'][:]
# valid_dataset=np.transpose(valid_dataset,(0,2,1))
# test_dataset=Data_a['test_dataset'][:]
# test_dataset=np.transpose(test_dataset,(0,2,1))
# #

# train_dataset=np.load(r"D:\pythonProject\train_dataset.npy")
# valid_dataset=np.load(r"D:\pythonProject\valid_dataset.npy")
# test_dataset=np.load(r"D:\pythonProject\test_dataset.npy")


# train_input=train_dataset[:,0:9,:]
train_input=train_dataset[:,0:2,:]
#训练输入数据集 (0.8*  区间是左闭右开  ndarray
train_output=train_dataset[:,2:5,:] #训练输出数据集  (0.8
valid_input=valid_dataset[:,0:2,:]
valid_output=valid_dataset[:,2:5,:]
# test_input=test_dataset[:,0:4,:]
# test_output=test_dataset[:,3:22,:]
train_num=len(train_input)
val_num = len(valid_input)


tr_dataset=mydata(train_input,train_output)
va_dataset=mydata(valid_input,valid_output)
# te_dataset=mydata(test_input,test_output)
# #
np.save('train_dataset.npy',train_dataset)
np.save('valid_dataset.npy',valid_dataset)
# np.save('test_dataset.npy',test_dataset)

# train_input1=np.pad(train_input,((0,0),(0,15),(0,0)),constant_values = (0,0))
# valid_input1=np.pad(valid_input,((0,0),(0,15),(0,0)),constant_values = (0,0))
# train_input1=np.reshape(train_input1,(train_input1[0],1,train_input1[1],train_input1[2]))
# valid_input1=np.reshape(valid_input1,(valid_input1[0],1,valid_input1[1],valid_input1[2]))
# tr_dataset=torch.from_numpy(train_input1),torch.from_numpy(train_output)
# va_dataset=torch.from_numpy(valid_input1),torch.from_numpy(valid_output)
batch_size = 64 ### 64
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

train_loader = torch.utils.data.DataLoader(tr_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0,pin_memory=True)
validate_loader = torch.utils.data.DataLoader(va_dataset,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=0,pin_memory=True)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
train_steps = len(train_loader)


net=myNet(init_weights=True)
# net=myNet()
# path=r"D:\pythonProject\model_logs\myNet_loss 4.228368197800592e-06_val_loss 6.989769767570246e-08.pth"
# checkpoint=torch.load(path)
# net.load_state_dict(checkpoint['net'])

torch.cuda.empty_cache()
net.to(device)
loss_function = nn.MSELoss(reduction='mean')
optimizer=optim.Adam(net.parameters(),lr=9e-5)
# optimizer=optim.SGD(net.parameters(),lr=0.1,momentum=0.9,dampening=0.5, weight_decay=0.01, nesterov=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                     milestones=[50,100, 140, 200], gamma=0.8)
# optimizer.load_state_dict(checkpoint['optimizer'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-09)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15,30],gamma = 0.2)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
# writer.add_graph(net, torch.Tensor(1,1,3, 5000))
epochs = 100
best_acc = 0.0
# train_steps = len(train_loader)
fr=100

total_step=0


# start_epoch=checkpoint['epoch']+1
start_epoch=0
for epoch in range(start_epoch,epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    # if epoch%30==0:
    #      for p in optimizer.param_groups:
    #          p['lr'] *=0.5


    for step, data in enumerate(train_loader):
        total_step=total_step+1
        input, output = data
        # input 两行加速度数据和两行应变数据 matlab已调整为同一数量级
        acc = input[:, :, 0:1, :]
        input[:,:,1:2,:]=input[:,:,1:2,:]*(-100)  ##简支梁： 应变 -100 转角 1  连续梁：应变 -100 转角 1
        input[:, :, 0:1, :] =input[:, :, 0:1, :]*(1e-3)   ###简支梁 加速度：1e-4 连续梁 加速度 1e-3

        ### 试验用的数据
        # input[:, :, 1:2, :] = input[:, :, 1:2, :] * (1e-3)
        # input[:, :, 0:1, :] = input[:, :, 0:1, :] * (1e-2)
        optimizer.zero_grad()

        displacement1_hi,displacement1_pseudo,displacement1 = net(input.to(device))
        # displacement1_hi, displacement1_pseudo1,displacement1_pseudo2, displacement1 = net(input.to(device))
        output_hi=output[:,:,0:1,:]
        output_pseudo=output[:,:,1:2,:]
        output_dis=output[:,:,2:3,:]

        loss_dis1 = loss_function(displacement1_hi, output_hi.to(device))
        loss_dis2= loss_function(displacement1_pseudo, output_pseudo.to(device))
        # loss_dis = loss_function(displacement1, output_dis.to(device))
        loss_dis=loss_function(displacement1, output_dis.to(device))
        # loss0 = (1e-4*loss_ac + loss_dis) #e-8~e-10
        # ###

        displacement_detach=displacement1.cpu().detach().numpy()
        # # #
        vel_g=fr*np.gradient(displacement_detach,axis=3)
        acc_g=fr*np.gradient(vel_g,axis=3)

        loss1_1= (1e-6)*loss_function(acc.to(device),torch.FloatTensor(acc_g).to(device)) #-2
        # output_dis_detach=output_dis.cpu().detach().numpy()
        # vel_g_groundtruth = fr * np.gradient(output_dis_detach, axis=3)
        # displacement_pseudo_detach = displacement1_pseudo.cpu().detach().numpy()
        # output_pseudo_detach=output_pseudo.cpu().detach().numpy()
        # # # #
        # vel_pseudo = fr * np.gradient(displacement_pseudo_detach, axis=3)
        # vel_pseudo_groundtruth= fr * np.gradient(output_pseudo_detach, axis=3)
        # loss1_2=loss_function(torch.FloatTensor(vel_g).to(device),torch.FloatTensor(vel_g_groundtruth).to(device))
        # loss1=(loss1_1)

        # loss=loss0+loss1
        # loss=loss_dis+loss_dis1+loss_dis2
        loss =  1*loss_dis1 + loss_dis2+(1*loss_dis +0*loss1_1)
        # loss =  loss_dis+loss_dis1 + loss_dis2+0.08*loss1_2

        # loss = loss_dis + loss_dis1 + loss_dis2+loss1_1+loss1_2
        loss.backward()

        # print("loss_:{}".format(net.conv3.weight.grad.data.mean()))

        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
        #                                                          epochs,

        if step % 100==0:
            print('迭代次数：{}，训练次数：{}，loss：{}'.format(epoch,step,loss.item()))
        writer.add_scalar("Train_Loss", loss.item(), total_step)

        if total_step==1:
            print('lossdis:{}'.format(loss_dis))
            print('lossdis1:{}'.format(loss_dis1))
            print('lossdis2:{}'.format(loss_dis2))
            print('loss1_1:{}'.format(loss1_1))
    print('lossdis:{}'.format(loss_dis))
    print('lossdis1:{}'.format(loss_dis1))
    print('lossdis2:{}'.format(loss_dis2))
    print('loss1_1:{}'.format(loss1_1))
    # scheduler.step()
# validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    total_loss=0
    with torch.no_grad():
        # val_bar = tqdm(validate_loader)
        for step,val_data in enumerate(validate_loader):
            val_input, val_output = val_data
            val_acc=val_input[:, :, 0:1, :]
            val_input[:, :, 1:2, :] = val_input[:, :, 1:2, :]*(-100)
            val_input[:, :, 0:1, :] = val_input[:, :, 0:1, :]*(1e-3)

            # val_input[:, :, 1:2, :] = val_input[:, :, 1:2, :] * (1e-3)
            # val_input[:, :, 0:1, :] = val_input[:, :, 0:1, :] * (1e-2)
            displacement2_hi,displacement2_pseudo,displacement2 = net(val_input.to(device))
            val_output_hi = val_output[:, :, 0:1, :]
            val_output_pseudo = val_output[:, :, 1:2, :]
            val_output_dis = val_output[:, :, 2:3, :]

            val_loss_dis1 = loss_function(displacement2_hi, val_output_hi.to(device))
            val_loss_dis2 = loss_function(displacement2_pseudo, val_output_pseudo.to(device))
            val_loss_dis = loss_function(displacement2, val_output_dis.to(device))
            # val_loss_dis = loss_function(displacement2, val_output_dis.to(device))
            # val_loss0 = (1e-4*val_loss_ac  + val_loss_dis)
            # #

            val_displacement_detach = displacement2.cpu().detach().numpy()
            val_vel_g = fr * np.gradient(val_displacement_detach, axis=3)
            val_acc_g = fr * np.gradient(val_vel_g, axis=3)

            val_loss1_1 = (1e-6)*loss_function(val_acc.to(device), torch.FloatTensor(val_acc_g).to(device))

            # val_loss1=(val_loss1_1)
            # val_output_dis_detach = val_output_dis.cpu().detach().numpy()
            # val_vel_g_groundtruth = fr * np.gradient(val_output_dis_detach, axis=3)
            # displacement_pseudo_detach2 = displacement2_pseudo.cpu().detach().numpy()
            # val_output_pseudo_detach = val_output_pseudo.cpu().detach().numpy()
            # # # #
            # vel_pseudo2 = fr * np.gradient(displacement_pseudo_detach2, axis=3)
            # vel_pseudo2_groundtruth = fr * np.gradient(val_output_pseudo_detach, axis=3)
            # val_loss1_2 = loss_function(torch.FloatTensor(val_vel_g).to(device),
            #                               torch.FloatTensor(val_vel_g_groundtruth).to(device))

            val_loss = 1*val_loss_dis1+val_loss_dis2+(1*val_loss_dis+0*val_loss1_1)

            total_loss=total_loss+val_loss.item()

    val_avg_loss = total_loss/val_num
    print('测试集loss:{}'.format(val_avg_loss))
    writer.add_scalar("Valid_Loss", val_avg_loss, epoch)
    save_path = './A_model_logs/试验_0/myNet_loss {}_val_loss {}.pth'
    state_dict={"net":net.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch}
    torch.save(state_dict, save_path.format(loss,val_avg_loss))


print('Finished Training')
writer.close()
