import torch
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
from datetime import datetime
from datasets.MultiMnistMooDataset import MultiMnistMooDataset

from models.MooModel import MultiLeNetR

import warnings
from metrics.AccMetric import AccMetric

import torch.nn.functional as F
from utils.timer import TimeRecorder
import threading
from torch.optim.lr_scheduler import StepLR
import random
from torchvision import datasets, transforms


random.seed(42)
class SaveBestModel(object):
    def __init__(self, save_path):
        self.best_accuracy_improve = 0
        self.save_path = save_path
        self.best_epoch = 0
        self.L_base = 0.0
        self.R_base = 0.0


    def savebestmodel(self, model, score1 ,score2, cur_epoch):
        if cur_epoch == 0:
            tmp = (score1+score2)/2
            self.L_base = tmp
            self.R_base = tmp
            self.best_accuracy_improve = 0.0
            self.best_epoch = 0
            torch.save(model.state_dict(), self.save_path)
            return self.best_epoch
        
        impro1 = (score1-self.L_base)/self.L_base
        impro2 = (score2-self.R_base)/self.R_base
        cur_improve = (impro1+impro2)/2

        if cur_improve > self.best_accuracy_improve:
            self.best_epoch = cur_epoch
            self.best_accuracy_improve = cur_improve
            torch.save(model.state_dict(), self.save_path)
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
    
def pacfun(cur_batch,batch_num,cur_epoch,epoch):
    p=1+2*(cur_epoch/epoch)
    pac = 20*(1-((p*cur_batch)/batch_num))
    return pac


def threadfun(loss_list,th_size1,model_params):
    gradients_dict={}
    lock = threading.Lock()
    def compute_gradients_thread(index):
        nonlocal gradients_dict
        try:
            loss = loss_list[index]
            gradients = grad2vec(torch.autograd.grad(loss, model_params, allow_unused=True,retain_graph=True))
            lock.acquire()
            gradients_dict[index] = gradients
            lock.release()
        except Exception as e:
            print(f"Exception in thread {index}: {e}")
    threads = []
    for i in range(th_size1):
        t = threading.Thread(target=compute_gradients_thread, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    try:
        glist = [gradients_dict[i] for i in range(th_size1)]
    except Exception as e:
        print("Error")
        print("loss_list:",loss_list)
    return glist


def train(model, optimizer,scheduler,cur_epoch,epoch,data_loader, device, log_interval=50):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    batch_num = len(loader)
    for cur_batch, (img, labels) in enumerate(loader):
        data_num = labels.size(0)
        task_num = labels.size(1)
        img = img.to(device)
        labels = labels.to(device)
        pred = model(img,None)
        datalevel_loss_list = [torch.nn.functional.cross_entropy(pred[task_idx],labels[:,task_idx],reduction='none') for task_idx in range(task_num)]
        grad_list=[[] for _ in range(task_num)]
        cossim_list=[[] for _ in range(task_num)]
        magsim_list=[[] for _ in range(task_num)]
        weight_cossim = [[] for _ in range(task_num)]
        weight_magsim = [[] for _ in range(task_num)]  
        task_cossim = [[] for _ in range(task_num)] 
        task_magsim = [[] for _ in range(task_num)] 
        model_params = list(model.parameters())
        # ------------
        task_pair_num = task_num-1

        th_size = 30
        th_nums = data_num//th_size
        for t in range(task_num):
            for i in range(th_nums):
                start_index = i * th_size
                end_index = (i + 1) * th_size
                losses = datalevel_loss_list[t][start_index:end_index]
                glist = threadfun(losses,th_size,model_params)
                grad_list[t]+=glist

            if data_num % th_size > 0:
                start_index = th_nums * th_size
                end_index = data_num
                last_size = end_index-start_index
                losses = datalevel_loss_list[t][start_index:end_index]
                glist = threadfun(losses,last_size,model_params)
                grad_list[t]+=glist
        

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

        loss_s = [1-(datalevel_loss_list[i])/loss_mean[i] for i in range(task_num)]

        weight_dataloss = [torch.sigmoid(loss_s[i].mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for i in range(task_num)]
        
        new_task_cossim = [torch.sign(tensor) * torch.sqrt(torch.abs(tensor)) for tensor in task_cossim]
        weight_cossim = [torch.sigmoid(x.mul(2.0).mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for x in new_task_cossim]

        new_task_magsim = [torch.sign(tensor) * torch.sqrt(torch.abs(tensor)) for tensor in task_magsim]
        weight_magsim = [torch.sigmoid((x-0.5).mul(2.0).mul(pacfun(cur_batch,batch_num,cur_epoch,epoch))).mul(2.0) for x in new_task_magsim]

        final_weight_n = [torch.sqrt((weight_dataloss[idx]+weight_cossim[idx]+weight_magsim[idx])/3) for idx in range(task_num)]


        sum_weight = [torch.sum(final_weight_n[idx]) for idx in range(task_num)]
        final_weight = [final_weight_n[idx]/sum_weight[idx] for idx in range(task_num)]

        weight_loss = [torch.unsqueeze(torch.sum((final_weight[idx]*datalevel_loss_list[idx])),dim=0) for idx in range(task_num)]       
        wloss = 0
        for item in weight_loss:            
            wloss += item
        optimizer.zero_grad()
        wloss.backward()
        optimizer.step()
        total_loss += wloss.item()

        if (cur_batch + 1) % log_interval == 0:
            # print('wloss',wloss)
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    # scheduler.step()
    
def test(model, data_loader, task_num, device):
    model.eval()
    LtestAccMetric = AccMetric()
    RtestAccMetric = AccMetric()
    testloader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    test_loss = {}
    loss_results = list()
    for i in range(task_num):
        test_loss[i]=list()
    with torch.no_grad():
        for cur_batch, (img, labels) in enumerate(testloader):
            img = img.to(device)
            labels = labels.to(device)
            pred = model(img,None)
            for task_idx in range(task_num):
                test_loss[task_idx].extend(torch.nn.functional.cross_entropy(pred[task_idx],labels[:,task_idx],reduction='none'))
            LtestAccMetric.update_fun(pred=pred[0],gt=labels[:,0])
            RtestAccMetric.update_fun(pred=pred[1],gt=labels[:,1])
            # break
    for i in range(task_num):
        loss_results.append(torch.mean(torch.stack(test_loss[i])).item())
    L_score = LtestAccMetric.score_fun()
    R_score = RtestAccMetric.score_fun()
    return L_score, R_score, loss_results


def main(data_path, task_num, epoch, learning_rate, 
         train_batch_size, test_batch_size, weight_decay, device, save_dir):
    # ------
    gpus = [1,2]
    device = torch.device(device)
    # ------
    # data_path /home/admin/LiuZeYu/curriculum/data/multi-mnist
    transformer=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MultiMnistMooDataset(root=data_path,train=True,transform=transformer,target_transform=None,multi=True)
    test_dataset = MultiMnistMooDataset(root=data_path,train=False,transform=transformer,target_transform=None,multi=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    model = MultiLeNetR().to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpus)
    # -----------
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = None
    now=datetime.now()
    timestamp = datetime.timestamp(now)
    current_time = datetime.fromtimestamp(timestamp)
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(f'{save_dir}/{time_str}')
    dataset_name = 'MultiMnist'
    save_path=f'{save_dir}/{time_str}/{dataset_name}.pt'
    best_model = SaveBestModel(save_path=save_path)
    best_epoch=0

    for epoch_i in range(epoch):
        train(model, optimizer,scheduler,epoch_i,epoch, train_loader, device)
        L_score,R_score, loss = test(model, test_loader, task_num, device)
        print('L_score',L_score)
        print('R_score',R_score)
        # print('lr:',optimizer.param_groups[0]['lr'])
        print('epoch:', epoch_i, 'L_Acc score:', L_score, 'R_Acc score:,',R_score)
        for i in range(task_num):
            print('task {}, loss {}'.format(i, loss[i]))
        best_epoch = best_model.savebestmodel(model,L_score,R_score,cur_epoch=epoch_i)
        print('current best epoch:',best_epoch)


    model.load_state_dict(torch.load(save_path))
    L_score,R_score, loss = test(model, test_loader, task_num, device)
    f = open('{}.txt'.format(dataset_name), 'a', encoding = 'utf-8')
    f.write('\n')
    f.write('time:{}\n'.format(time_str))
    f.write('learning rate: {}\n'.format(learning_rate))
    print('best epoch {}'.format(best_epoch))
    print('best ', 'L_score:', L_score, 'R_score',R_score)
    for i in range(task_num):
        print('task {},  Log-loss {}'.format(i, loss[i]))
        f.write('task {}, Log-loss {}\n'.format(i, loss[i]))
    f.write('best epoch {}, L_score {}, R_score {}\n'.format(best_epoch, L_score, R_score))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/admin/LiuZeYu/curriculum/data/multi-mnist')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=1.0e-4)
    # ---------------------------------------------------------
    parser.add_argument('--train_batch_size', type=int, default=64) # 2048
    parser.add_argument('--test_batch_size', type=int, default=128) # 2048
    # ---------------------------------------------------------
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--save_dir', default='./chkpt')
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
    args = parser.parse_args()
    main(args.data_path,
         args.task_num,
         args.epoch,
         args.learning_rate,
         args.train_batch_size,
         args.test_batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)