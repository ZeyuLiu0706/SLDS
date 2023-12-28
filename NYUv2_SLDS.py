# NYU
import torch
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
from datetime import datetime
from datasets.NYUv2Dataset import NYUv2Dataset
import models.nyuv2model
import models.encoder.resnet_dilated
import models.decoder.DeepLabHead
from loss.SegLoss import SegLoss
from loss.DepthLoss import DepthLoss
from loss.NormalLoss import NormalLoss
from metrics.SegMetric import SegMetric
from metrics.DepthMetric import DepthMetric
from metrics.NormalMetric import NormalMetric
import torch.nn.functional as F
from utils.timer import TimeRecorder
import threading
import math
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import random

random.seed(777)
class SaveBestModel(object):
    def __init__(self, save_path):
        self.best_accuracy_improve = 0
        self.save_path = save_path
        self.best_epoch = 0
        self.seg_list = ['mIoU','pixAcc']
        self.dep_list = ['abs_err', 'rel_err']
        self.nor_list = ['mean', 'median', '<11.25', '<22.5', '<30'] 
        self.base_seg = {}
        self.base_dep = {}
        self.base_nor = {}
        self.best_seg = {}
        self.best_dep = {}
        self.best_nor = {}
        for mn in self.seg_list:
            self.base_seg[mn]=0.0
        for mn in self.dep_list:
            self.base_dep[mn]=0.0
        for mn in self.nor_list:
            self.base_nor[mn]=0.0

    def savebestmodel(self, model, score1 ,score2 ,score3, cur_epoch):
        if cur_epoch == 0:
            for mn in self.seg_list:
                self.base_seg[mn]=score1[mn]
            for mn in self.dep_list:
                self.base_dep[mn]=score2[mn]
            for mn in self.nor_list:
                self.base_nor[mn]=score3[mn]
            self.best_accuracy_improve = 0.0
            self.best_epoch = 0
            state={}
            for m in model:
                state[m] = model[m].state_dict()
            torch.save(state, self.save_path)
            return self.best_epoch
        
        impro1 = sum([(score1[mn]-self.base_seg[mn])/self.base_seg[mn] for mn in self.seg_list])/2
        
        impro2 = sum([(self.base_dep[mn]-score2[mn])/self.base_dep[mn] for mn in self.dep_list])/2
        
        impro_tmp1 = (self.base_nor['mean']-score3['mean'])/self.base_nor['mean'] +\
              (self.base_nor['median']-score3['median'])/self.base_nor['median']
        
        impro_tmp2 = ((score3['<11.25']-self.base_nor['<11.25'])/self.base_nor['<11.25'])+\
            ((score3['<22.5']-self.base_nor['<22.5'])/self.base_nor['<22.5'])+\
                ((score3['<30']-self.base_nor['<30'])/self.base_nor['<30'])
        
        impro3 = (impro_tmp1+impro_tmp2)/len(self.nor_list)
        
        cur_improve = (impro1+impro2+impro3)/3

        if cur_improve > self.best_accuracy_improve:
            self.best_epoch = cur_epoch
            self.best_accuracy_improve = cur_improve
            self.best_seg=score1
            self.best_dep=score2
            self.best_nor=score3
            state={}
            for m in model:
                state[m] = model[m].state_dict()
            torch.save(state, self.save_path)
        return self.best_epoch

def grad2vec(origin_grad):
    return torch.cat([grad.flatten() for grad in origin_grad if grad is not None])

def cos_sim(grad1,grad2):
    if grad1.size(0) != grad2.size(0):
        size = max(grad1.size(0), grad2.size(0))
        gap = abs(grad1.size(0) - grad2.size(0))
        if grad1.size(0) == size:
            grad2 = torch.cat([grad2, torch.zeros(gap).to(grad2.device)])
        elif grad2.size(0) == size:
            grad1 = torch.cat([grad1, torch.zeros(gap).to(grad1.device)])
        grad1 = grad1.view(size, -1)
        grad2 = grad2.view(size, -1)
    return (F.cosine_similarity(grad1, grad2, dim=0)).squeeze()

def magnitude_sim(grad1,grad2):
    grad1_mag = torch.norm(grad1)
    grad2_mag = torch.norm(grad2)
    tmp1 = 2*grad1_mag*grad2_mag
    tmp2 = torch.square(grad1_mag)+torch.square(grad2_mag)
    return tmp1/tmp2
 
def record_weight(datalevel_loss_list,task_cossim,task_magsim,final_weight,cur_batch):
    we = open("nyu_weight_recoder.txt",'a',encoding='utf-8')
    we.write('cur_batch:{}\n'.format(cur_batch))
    we.write('datalevel_loss_list {}\n task_cossim {}\n task_magsim {}\n final_weight {}\n '.format(datalevel_loss_list, task_cossim, task_magsim,final_weight))
    we.write('-----------------------------------------------\n')
    we.close()
 
   
def pacfun(cur_batch,batch_num,cur_epoch,epoch):
    p = 1+cur_epoch/epoch
    pac = 20*math.exp(-(p*cur_batch/batch_num))
    return pac


def train(model, optimizer,scheduler,cur_epoch ,epoch,c_loss,data_loader, device, log_interval=100):
    for m in model:
        model[m].train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    batch_num = len(loader)
    segloss = SegLoss()
    deploss = DepthLoss()
    norloss = NormalLoss()
    for cur_batch, (img, labels) in enumerate(loader):
        task_num = len(labels)
        data_num = len(labels[0])
        img = img.to(device)
        encoder_output = model[0](img)
        pred = [ [] for _ in range(task_num) ]
        for i in range(task_num):
            pred[i] = model[i+1](encoder_output)
        seg_loss = segloss.compute_loss(pred=pred[0], gt=labels[0].to(device))
        seg_loss_list = [torch.mean(seg_loss[i]) for i in range(data_num)]
        seg_loss_list = torch.cat([tensor.unsqueeze(0) for tensor in seg_loss_list], dim=0)
        dep_loss_list = deploss.compute_loss(pred=pred[1], gt=labels[1].to(device))
        dep_loss_list = torch.cat([tensor.unsqueeze(0) for tensor in dep_loss_list], dim=0)
        nor_loss_list = norloss.compute_loss(pred=pred[2],gt=labels[2].to(device))
        nor_loss_list = torch.cat([tensor.unsqueeze(0) for tensor in nor_loss_list], dim=0)
        datalevel_loss_list = [seg_loss_list,dep_loss_list,nor_loss_list]
        grad_list=[[] for _ in range(task_num)]
        cossim_list=[[] for _ in range(task_num)]
        magsim_list=[[] for _ in range(task_num)]
        weight_cossim = [[] for _ in range(task_num)]
        weight_magsim = [[] for _ in range(task_num)]  
        task_cossim = [[] for _ in range(task_num)] 
        task_magsim = [[] for _ in range(task_num)] 
        en_model_params = list(model[0].parameters())
        # ------------
        task_pair_num = task_num-1
        # caculate gradient 
        for t in range(task_num):
            grad_list[t] = [grad2vec(torch.autograd.grad(datalevel_loss_list[t][i], en_model_params, allow_unused=True,retain_graph=True)) for i in range(data_num)]
        # caculate similarity
        for task_idx1 in range(task_num):
            for task_idx2 in range(task_idx1+1,task_num):
                cossim = [cos_sim(grad_list[task_idx1][i],grad_list[task_idx2][i]) for i in range(data_num)]
                magsim = [magnitude_sim(grad_list[task_idx1][i],grad_list[task_idx2][i]) for i in range(data_num)]
                cossim_list[task_idx1].append(cossim)
                cossim_list[task_idx2].append(cossim)
                magsim_list[task_idx1].append(magsim)
                magsim_list[task_idx2].append(magsim)
        for idx in range(task_num):
            cossim_tensor_matrix = torch.stack([torch.stack(row) for row in cossim_list[idx]])
            magsim_tensor_matrix = torch.stack([torch.stack(row) for row in magsim_list[idx]])
     
            task_cossim[idx] = (cossim_tensor_matrix.sum(dim=0))/task_pair_num
            task_magsim[idx] = (magsim_tensor_matrix.sum(dim=0))/task_pair_num
        loss_mean = [tensor.mean() for tensor in datalevel_loss_list]
        loss_dis = [(loss_mean[i]-datalevel_loss_list[i])/loss_mean[i] for i in range(task_num)]
        weight_dataloss = [torch.sigmoid((loss_dis[i]).mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for i in range(task_num)]

        weight_cossim = [torch.sigmoid(x.mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for x in task_cossim]

        weight_magsim = [torch.sigmoid((x-0.5).mul(2.0).mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for x in task_magsim]

        weight_lo_sum = [torch.sum(weight_dataloss[idx]) for idx in range(task_num)]
        weight_dataloss = [weight_dataloss[idx]/weight_lo_sum[idx] for idx in range(task_num)]
        
        weight_ang_sum = [torch.sum(weight_cossim[idx]) for idx in range(task_num)]
        weight_cossim = [weight_cossim[idx]/weight_ang_sum[idx] for idx in range(task_num)]

        weight_mag_sum = [torch.sum(weight_magsim[idx]) for idx in range(task_num)]
        weight_magsim = [weight_magsim[idx]/weight_mag_sum[idx] for idx in range(task_num)]

        final_weight_n = [(weight_dataloss[idx]+weight_cossim[idx]+weight_magsim[idx])/3 for idx in range(task_num)]

        weight_loss = [torch.unsqueeze(torch.sum((final_weight_n[idx]*datalevel_loss_list[idx])),dim=0) for idx in range(task_num)]       
        wloss = 0
        wloss = (weight_loss[0]+weight_loss[1]+weight_loss[2])

        optimizer.zero_grad()
        wloss.backward()
        optimizer.step()
        total_loss += wloss.item()
        if (cur_batch + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    scheduler.step()
    


def test(model, data_loader, task_num, device):
    for m in model:
        model[m].eval()
    testsegloss = SegLoss()
    testdeploss = DepthLoss()
    testnorloss = NormalLoss()
    testSegMetric = SegMetric(num_classes=13)
    testDepthMetric = DepthMetric()
    testNormalMetric = NormalMetric()
    testloader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    test_loss = [[] for _ in range(task_num)]
    test_seg_score = {}
    test_dep_score = {}
    test_nor_score = {}
    seg_list = ['mIoU','pixAcc']
    dep_list = ['abs_err', 'rel_err']
    nor_list = ['mean', 'median', '<11.25', '<22.5', '<30']
    for mn in seg_list:
        test_seg_score[mn]=0.0
    for mn in dep_list:
        test_dep_score[mn]=0.0
    for mn in nor_list:
        test_nor_score[mn]=0.0
    with torch.no_grad():
        for cur_batch, (img, labels) in enumerate(testloader):
            img = img.to(device)
            encoder_output = model[0](img)
            pred = [ [] for _ in range(task_num) ]
            for i in range(task_num):
                pred[i] = model[i+1](encoder_output)
            test_loss[0].append(testsegloss.test_loss(pred=pred[0],gt=labels[0].to(device)))
            test_loss[1].append(testdeploss.test_loss(pred=pred[1],gt=labels[1].to(device)))
            test_loss[2].append(testnorloss.test_loss(pred=pred[2],gt=labels[2].to(device)))
            testSegMetric.update_fun(pred=pred[0],gt=labels[0].to(device))
            testDepthMetric.update_fun(pred=pred[1],gt=labels[1].to(device))
            testNormalMetric.update_fun(pred=pred[2],gt=labels[2].to(device))  
    loss_results = [torch.cat([tensor.unsqueeze(0) for tensor in test_loss[t]], dim=0) for t in range(task_num)]
    loss_results = [torch.mean(t) for t in loss_results]
    test_seg_score = testSegMetric.score_fun()
    test_dep_score = testDepthMetric.score_fun()
    test_nor_score = testNormalMetric.score_fun()
    return test_seg_score, test_dep_score, test_nor_score, loss_results


def main(data_path, task_num, model_name, epoch, learning_rate, 
         train_batch_size, test_batch_size, weight_decay, device, save_dir):
    device = torch.device(device)
    train_dataset = NYUv2Dataset(data_path+'/train')
    test_dataset = NYUv2Dataset(data_path+'/val')
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2, shuffle=False)
    encoder = models.encoder.resnet_dilated.resnet_dilated(basenet='resnet50').to(device)
    decoder1 = models.decoder.DeepLabHead.DeepLabHead(input_channels=2048,output_channels=13,img_size=[288, 384]).to(device)
    decoder2 = models.decoder.DeepLabHead.DeepLabHead(input_channels=2048,output_channels=1,img_size=[288, 384]).to(device)
    decoder3 = models.decoder.DeepLabHead.DeepLabHead(input_channels=2048,output_channels=3,img_size=[288, 384]).to(device)
    model={}
    model[0] = encoder
    model[1] = decoder1
    model[2] = decoder2
    model[3] = decoder3

    model_params = []
    for m in model:
        model_params += model[m].parameters()

    optimizer = torch.optim.Adam(params=model_params, lr=learning_rate, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=35, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=200,eta_min=3e-5)

    now=datetime.now()
    timestamp = datetime.timestamp(now)
    current_time = datetime.fromtimestamp(timestamp)
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(f'{save_dir}/{time_str}')
    dataset_name = 'nyuv2'
    save_path=f'{save_dir}/{time_str}/{dataset_name}_{model_name}.pt'
    best_model = SaveBestModel(save_path=save_path)
    best_epoch=0
    c_loss = [0,0,0]
    for epoch_i in range(epoch):
        train(model, optimizer,scheduler,epoch_i,epoch,c_loss,train_data_loader, device)
        seg_score, dep_score , nor_score, loss = test(model, test_data_loader, task_num, device)
        c_loss = loss
        print('lr:',optimizer.param_groups[0]['lr'])
        print('epoch:', epoch_i, 'seg_score:', seg_score, 'dep_score:,',dep_score,'nor_score:', nor_score)
        for i in range(task_num):
            print('task {}, loss {}'.format(i, loss[i]))
        best_epoch = best_model.savebestmodel(model,seg_score,dep_score,nor_score,cur_epoch=epoch_i)
        print('current best epoch:',best_epoch)
    seg_score, dep_score , nor_score=best_model.best_seg, best_model.best_dep, best_model.best_nor
    f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding = 'utf-8')
    f.write('\n')
    f.write('time:{}\n'.format(time_str))
    f.write('learning rate: {}\n'.format(learning_rate))
    print('best epoch {}'.format(best_epoch))
    print('best ', 'seg_score:', seg_score, 'dep_score',dep_score, 'nor_score',nor_score)
    for i in range(task_num):
        print('task {},  Log-loss {}'.format(i, loss[i]))
        f.write('task {}, Log-loss {}\n'.format(i, loss[i]))
    f.write('best epoch {}/{}, seg_score {}, dep_score {}, nor_score {}\n'.format(best_epoch, epoch, seg_score, dep_score, nor_score))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/nyuv2')
    parser.add_argument('--model_name', default='sharedbottom', choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'])
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)
    parser.add_argument('--weight_decay', type=float, default=1.0e-5)
    # ---------------------------------------------------------
    parser.add_argument('--train_batch_size', type=int, default=4) # 2048
    parser.add_argument('--test_batch_size', type=int, default=8) # 2048
    # ---------------------------------------------------------
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--save_dir', default='./chkpt')
    args = parser.parse_args()
    main(args.data_path,
         args.task_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.train_batch_size,
         args.test_batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)