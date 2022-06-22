from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import numpy as np
from model_ac_strain import myNet
from torch.utils.tensorboard import SummaryWriter
import h5py

import pandas as pd
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

#
data_root=r'D:\phycnn_ac\train'

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# dataname=os.path.join(data_root,'te_dataset2.mat')
dataname=os.path.join(data_root,'te_dataset_u1_s.mat') #te_dataset_test_midspan_3
Data_a=h5py.File(dataname)               #读取mat文件
# Data = Data_a['te_dataset'][:]
Data = Data_a['tr_dataset'][:]
test_dataset=np.transpose(Data,(0,2,1)) #
# test_dataset=np.load(dataname)
# print("test:{}".format(len(test_dataset)))

test_input=test_dataset[:,0:2,:]
test_output=test_dataset[:,2:5,:]
te_dataset=mydata(test_input,test_output)
test_loader = torch.utils.data.DataLoader(te_dataset,batch_size=1, shuffle=False,num_workers=0)
# writer=SummaryWriter('logs')
# pretrained=True就可以使用预训练的模型
net=myNet()

# writer.add_graph(net, torch.Tensor(1,1,3, 5000))

# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.8586719306767918e-05_val_loss 1.784516224354604e-05.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.3146261153451633e-05_val_loss 4.571367422675128e-06.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.2308737495914102e-05_val_loss 5.379647192190381e-06.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.5429877748829313e-06_val_loss 7.540939184647044e-07.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.2148728274041787e-05_val_loss 6.329192870424549e-06.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 4.295652615837753e-06_val_loss 2.5357571239818335e-06.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\连续_转角_无噪声_phy0.05\myNet_loss 6.824235470048734e-07_val_loss 6.767684231986e-07.pth'
# pthfile = r'D:\phycnn_ac\model_logs\myNet_loss 1.1954236356359615e-07_val_loss 1.19974800575721e-07.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\myNet_loss 3.360506525496021e-06_val_loss 3.3661121791557965e-06.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\连续_应变_加噪声_phy0.05\myNet_loss 3.3757826258806745e-06_val_loss 3.3890822773940955e-06.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\连续_应变_无噪声_phy0.05\myNet_loss 2.2730171167495428e-07_val_loss 2.3366034016906548e-07.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\连续_应变_无噪声_phy0.05_Node6_边跨\myNet_loss 9.327931849156812e-08_val_loss 9.320247865152176e-08.pth'

################################

# pthfile = r'D:\phycnn_ac\A_model_logs\简支_应变_无噪声_phy0.05\myNet_loss 1.8334618800963653e-07_val_loss 1.8845270058909235e-07.pth'

# pthfile = r'D:\phycnn_ac\A_model_logs\简直_应变_加噪声_无phy\myNet_loss 1.278477611776907e-05_val_loss 1.27798570796737e-05.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\简支_应变_加噪声_phy0.05\myNet_loss 1.2841091120208148e-05_val_loss 1.2793484096107477e-05.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\简支_应变_加噪声_phy0.5\myNet_loss 1.300425356021151e-05_val_loss 1.2856260161697719e-05.pth'

# pthfile = r'D:\phycnn_ac\A_model_logs\简支梁_转角_无噪声_phy0.05\myNet_loss 1.2959708328708075e-07_val_loss 1.2643739891737837e-07.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\简支梁_应变_无噪声_phy0.05_Node10\myNet_loss 1.9863368549977167e-07_val_loss 1.9076973470921743e-07.pth'

# pthfile = r'D:\phycnn_ac\A_model_logs\简支梁_转角_无噪声_phy0.05\myNet_loss 2.065013632090995e-06_val_loss 1.969237289382401e-06.pth'    #epoch=1
#pthfile = r'D:\phycnn_ac\A_model_logs\简支梁_转角_无噪声_phy0.05\myNet_loss 1.6334391261807468e-07_val_loss 1.6509084116804e-07.pth'     #epoch=50
# pthfile = r'D:\phycnn_ac\A_model_logs\简支梁_转角_无噪声_phy0.05\myNet_loss 1.2792528991667496e-07_val_loss 1.3172201404065831e-07.pth'   #epoch=100
# pthfile = r'D:\phycnn_ac\A_model_logs\myNet_loss 0.0036713331937789917_val_loss 0.0036873356400083826.pth'
# pthfile = r'D:\phycnn_ac\A_model_logs\myNet_loss 0.030560098588466644_val_loss 0.041449004144440256.pth'

# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 0.013111512176692486_val_loss 0.02123918057180125.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 0.0008957835962064564_val_loss 0.0007474717428420948.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 2.403667531325482e-05_val_loss 3.186074500841366e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 2.3430027795257047e-05_val_loss 2.6712520386914672e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 2.3430027795257047e-05_val_loss 2.6712520386914672e-05.pth'






# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 2.335287535970565e-05_val_loss 3.075014887776321e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\myNet_loss 3.144349466310814e-05_val_loss 3.3057804666162294e-05.pth'

# pthfile=r'D:\phycnn_ac\A_model_logs\试验_test_sidespan\myNet_loss 0.00010590621968731284_val_loss 0.00011307979799347885.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0_仅应变噪声\myNet_loss 3.389844550838461e-06_val_loss 3.3753213647476816e-06.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0.05_仅应变噪声\myNet_loss 3.3982214517891407e-06_val_loss 3.3835650363300827e-06.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0.5_仅应变噪声\myNet_loss 3.589154403016437e-06_val_loss 3.5887801416834007e-06.pth'

# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0.0005_仅应变噪声\myNet_loss 3.3828691812232137e-06_val_loss 3.3771951645888974e-06.pth'


# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0.05_应变加噪位移不加噪声\myNet_loss 4.397473958306364e-07_val_loss 4.1682200052258386e-07.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0.5_应变加噪位移不加噪声\myNet_loss 6.701345682813553e-07_val_loss 6.681912448976609e-07.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\连续梁_噪声_0_应变加噪位移不加噪声\myNet_loss 4.172864578322333e-07_val_loss 4.198301180628287e-07.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\试验_0\myNet_loss 3.597439354052767e-05_val_loss 3.669110786411644e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\试验_0\myNet_loss 2.1504645701497793e-05_val_loss 2.029585469284948e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\试验_0\myNet_loss 2.164008401450701e-05_val_loss 2.3823293723471755e-05.pth'
# pthfile=r'D:\phycnn_ac\A_model_logs\试验_0\myNet_loss 2.067007517325692e-05_val_loss 1.9058927126098464e-05.pth'

# pthfile=r'D:\phycnn_ac\A_model_logs\试验_0\myNet_loss 2.981368538712559e-07_val_loss 2.889254789898802e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy1e-7\myNet_loss 2.473700533300871e-07_val_loss 2.4472839200336456e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy1e-8\myNet_loss 2.1211033640611276e-07_val_loss 2.0912728125506891e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy1e-7\myNet_loss 2.7664756885315e-07_val_loss 2.6996789605308416e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_nophy\myNet_loss 1.844876464929257e-07_val_loss 1.8995322140225035e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy5e-8\myNet_loss 1.8782978372655634e-07_val_loss 2.010324528104827e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy5e-9\myNet_loss 1.9935852435537527e-07_val_loss 1.999406229432251e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy2e-8\myNet_loss 1.9000863460405526e-07_val_loss 1.9513946952394908e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy1e-8\myNet_loss 2.3536809123925195e-07_val_loss 2.343038260632646e-07.pth'



# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy2e-8\myNet_loss 2.3653419134461728e-07_val_loss 2.467144834280196e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy5e-8\myNet_loss 2.3445228691798548e-07_val_loss 2.370407559163882e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy1e-7\myNet_loss 4.111543887574953e-07_val_loss 3.3388742756527513e-07.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_nophy\myNet_loss 2.529540097384597e-07_val_loss 2.507299870058155e-07.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy3e-8\myNet_loss 2.884911225464748e-07_val_loss 2.831626626953465e-07.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy4e-8\myNet_loss 2.8825203912674624e-07_val_loss 2.941682150166732e-07.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy2e-8\myNet_loss 3.3923122373380465e-06_val_loss 3.3824707627398234e-06.pth'


# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy1e-7\myNet_loss 3.3640526453382336e-06_val_loss 3.385608078332325e-06.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_nophy\myNet_loss 3.4638958368304884e-06_val_loss 3.460589925789274e-06.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy2e-8\myNet_loss 3.463694611127721e-06_val_loss 3.4313497077202254e-06.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy5e-7\myNet_loss 3.7574611724267015e-06_val_loss 3.761234608745667e-06.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy2e-8\myNet_loss 3.3572428037587088e-06_val_loss 3.3782652574624685e-06.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_噪声_phy5e-8\myNet_loss 3.332552751089679e-06_val_loss 3.35012364800491e-06.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\试验_phy1e-8\myNet_loss 2.3746562874293886e-05_val_loss 2.5227575924584154e-05.pth'
# pthfile = r'D:\phycnn_ac\新建文件夹 (5)\试验_phy1e-9\myNet_loss 1.9463093849481083e-05_val_loss 2.1184579553477426e-05.pth'
# pthfile = r'D:\phycnn_ac\新建文件夹 (5)\试验_phy1e-10\myNet_loss 1.341209099336993e-05_val_loss 1.8057520882480598e-05.pth'

#pthfile = r'D:\phycnn_ac\新建文件夹 (5)\连续梁_转角_phy2e-8\myNet_loss 7.996123372322472e-07_val_loss 7.601042260940716e-07.pth'




# pthfile = r'D:\phycnn_ac\新建文件夹 (5)\连续梁_应变_phy2e-8_Node6\myNet_loss 4.370246386997678e-08_val_loss 4.310555340632701e-08.pth'
# pthfile = r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_1e-8_Node10\myNet_loss 1.9921513683129888e-07_val_loss 2.0263925974983722e-07.pth'
#pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy1e-8\myNet_loss 3.7204642922006315e-07_val_loss 3.623181316081414e-07_epoch50.pth'

# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_phy1e-8\myNet_loss 3.2116856800712412e-06_val_loss 3.2165561743606e-06_epoch2.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_nophy\myNet_loss 1.4780955552851083e-06_val_loss 1.4066555287387909e-06_e1.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_nophy\myNet_loss 4.604472110258939e-07_val_loss 4.4282166113962516e-07_e10.pth'
# pthfile=r'D:\phycnn_ac\新建文件夹 (5)\简支梁_应变_nophy\myNet_loss 2.0720207771773858e-07_val_loss 2.0715562215241808e-07_e100.pth'



pthfile=r'D:\phycnn_ac\A_model_logs\简支_应变_无噪声_phy0.05\myNet_loss 4.111110740723234e-07_val_loss 4.476556274409423e-07.pth'



checkpoint=torch.load(pthfile)
net.load_state_dict(checkpoint['net'])
# print(net)
net.to(device)

fr=100

total_step=0
loss_function = nn.MSELoss(reduction='mean')

test_num = len(test_input)
total_loss=0
total_loss1=0
total_loss2=0
total_loss3=0
total_loss1_1=0
# total_lossd=0
# total_loss1=0
# total_loss2=0

net.eval()
with torch.no_grad():
    for step, val_data in enumerate(test_loader):
        val_input, val_output = val_data
        val_acc = val_input[:, :, 0:1, :]

        val_input[:, :, 1:2, :] = val_input[:, :, 1:2, :] * (-100)
        val_input[:, :, 0:1, :] = val_input[:, :, 0:1, :] * (1e-4)

        # val_input[:, :, 1:2, :] = val_input[:, :, 1:2, :] * (1e-3)
        # val_input[:, :, 0:1, :] = val_input[:, :, 0:1, :] * (1e-2)
        displacement2_hi, displacement2_pseudo, displacement2 = net(val_input.to(device))
        val_output_hi = val_output[:, :, 0:1, :]
        val_output_pseudo = val_output[:, :, 1:2, :]
        val_output_dis = val_output[:, :, 2:3, :]

        val_loss_dis1 = loss_function(displacement2_hi, val_output_hi.to(device))
        val_loss_dis2 = loss_function(displacement2_pseudo, val_output_pseudo.to(device))
        val_loss_dis = loss_function(displacement2, val_output_dis.to(device))
        # val_loss_dis = loss_function(displacement2, val_output_dis.to(device))
        # val_loss0 = (1e-4*val_loss_ac  + val_loss_dis)
        # #

        # val_displacement_detach = displacement2.cpu().detach().numpy()
        # val_vel_g = fr * np.gradient(val_displacement_detach, axis=3)
        # val_acc_g = fr * np.gradient(val_vel_g, axis=3)
        val_vel_g = fr*torch.gradient(displacement2, dim=3)[0]
        val_acc_g = fr*torch.gradient(val_vel_g, dim=3)[0]
        # val_loss1_1 = (1e4*1e-4) * loss_function(val_acc.to(device), torch.FloatTensor(val_acc_g).to(device))


        # val_loss1_1 = (1e-8) * loss_function(val_acc_g.to(device), val_acc.to(device))


        # val_loss1=(val_loss1_1)
        # val_output_dis_detach = val_output_dis.cpu().detach().numpy()
        # val_vel_g_groundtruth = fr * np.gradient(val_output_dis_detach, axis=3)
        # displacement_pseudo_detach2 = displacement2_pseudo.cpu().detach().numpy()
        # val_output_pseudo_detach = val_output_pseudo.cpu().detach().numpy()
        # # # #
        # vel_pseudo2 = fr * np.gradient(displacement_pseudo_detach2, axis=3)
        # vel_pseudo2_groundtruth = fr * np.gradient(val_output_pseudo_detach, axis=3)
        # val_loss1_2 = loss_function(torch.FloatTensor(val_vel_g).to(device),
        #                             torch.FloatTensor(val_vel_g_groundtruth).to(device))

        val_loss = val_loss_dis1 + val_loss_dis2 + val_loss_dis#+val_loss1_1

        total_loss = total_loss + val_loss.item()
        total_loss1 = total_loss1 + val_loss_dis1.item()
        total_loss1_1=total_loss1_1+val_loss_dis.item()
        total_loss2 = total_loss2 + val_loss_dis2.item()
        # total_loss3 = total_loss3 + val_loss1_1.item()
        # total_lossa = total_lossa + val_loss_ac.item()
        # total_lossv = total_lossv + val_loss_ve.item()
        # total_lossd = total_lossd + val_loss_dis.item()
        # total_loss1 = total_loss1 + val_loss1_1.item()
        # total_loss2 = total_loss2 + val_loss1_2.item()
        #
        val_input_s = val_input.cpu().numpy()
        val_output_s = val_output.cpu().numpy()
        displacement2_hi=displacement2_hi.cpu().numpy()
        displacement2_pseudo=displacement2_pseudo.cpu().numpy()
        displacement2=displacement2.cpu().numpy()



        sa = np.concatenate((displacement2_hi, displacement2_pseudo, displacement2), axis=2)
        # sa=acceleration2_s

        val_input[:, :, 1, :] = val_input[:, :, 1, :] * (-1e-2)
        val_input[:, :, 0, :] = val_input[:, :, 0, :] * (1e4)
        val_input_s=np.reshape(val_input_s,(val_input_s.shape[2],val_input_s.shape[3]))
        val_output_s = np.reshape(val_output_s, (val_output_s.shape[2], val_output_s.shape[3]))
        sa = np.reshape(sa, (sa.shape[2], sa.shape[3]))
        save = np.concatenate((val_input_s, val_output_s, sa),axis=0)
        # val_output_dis_s=val_output_dis.cpu().numpy()
        # displacement2_s=displacement2.cpu().numpy()
        # save=np.concatenate((val_output_dis_s,displacement2_s),axis=2)
        # save=np.reshape(save,(save.shape[2],save.shape[3]))
        save_pd=pd.DataFrame(save)
        # save_pd.to_csv('./test_csv_c/simple_strain_no_noise_phy0.05_epoch100/{} testcsv.csv'.format(step), encoding='gbk')
        save_pd.to_csv('./test_csv_new/50/{} testcsv.csv'.format(step), encoding='gbk')


    val_avg_loss = total_loss / test_num
    val_avg_loss1 = total_loss1 / test_num
    val_avg_loss2 = total_loss2 / test_num
    val_avg_loss3 = total_loss3 / test_num
    val_avg_loss1_1=total_loss1_1/test_num
    # val_avg_lossd = total_lossd / test_num
    # val_avg_loss1 = total_loss1 / test_num
    # val_avg_loss2 = total_loss2 / test_num

print("test集的loss:{}".format(val_avg_loss))
print('loss1:{}'.format(val_avg_loss1))
print('loss2:{}'.format(val_avg_loss2))
print('loss1_1:{}'.format(val_avg_loss3))
print('loss_dis:{}'.format(val_avg_loss1_1))
# print('lossdis:{}'.format(val_avg_lossd))
# print('loss1_1:{}'.format(val_avg_loss1))
# print('loss1_2:{}'.format(val_avg_loss2))
# writer.close()
