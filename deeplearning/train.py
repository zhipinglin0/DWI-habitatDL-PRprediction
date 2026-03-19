
import os
import argparse
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import torchio as tio
from torchvision import transforms
from dataload.dataload3d import MyDataSet
import torch.optim.lr_scheduler as lr_scheduler

from model import Transformer3D
import sys
import time
from utils.lr_methods import warmup

from utils.train_engin import train_one_epoch, evaluate1
from datetime import datetime
now = datetime.now()
from tqdm import tqdm
from utils import create_lr_scheduler, get_params_groups, evaluate,evaluate_test

time = now.strftime("%Y-%m-%d-%H-%M-%S")
params = ''


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
parser.add_argument('--epochs', type=int, default=30, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.00001, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.01, help='end learning rate')
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--mm', type=float, default=0.9)
parser.add_argument('--seed', default=True, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision')
parser.add_argument('--name', type=str, default=time+'_')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--data_path_train', type=str,
                    default=r'./data')
parser.add_argument('--data_path_val', type=str,
                    default=r'./data')
parser.add_argument('--train_txt_path', type=str,
                    default=r'./3fold/train3.txt')  # change the training directory to ck19

parser.add_argument('--val_txt_path', type=str,
                    default=r'./3fold/val3.txt')  # change the training directory to ck19

opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)  # Python random module.
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
       
        print('random seed has been fixed')


    seed_torch()


def main(args):
    kkkkkk = 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if opt.tensorboard:
       
        log_path = os.path.join('./results/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path))

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) 

  
        tb_writer = SummaryWriter(log_path)

    data_path_train = args.data_path_train
    data_path_val = args.data_path_val


    train_txt_path=args.train_txt_path
    val_txt_path=args.val_txt_path
    best_auc = 0

    data_transform = {
        "train": tio.Compose([
            tio.ToCanonical(),  
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5), 
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  
            tio.RandomNoise(mean=0, std=0.1), 
            
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
            tio.RandomBlur(std=(0, 2)),  
            tio.RandomMotion(degrees=10, translation=10), 
            tio.RandomBiasField(coefficients=0.5),  
            tio.RescaleIntensity(out_min_max=(0, 1)),  
            tio.CropOrPad((128, 128, 128)), 
        ]),

        "val": tio.Compose([
            tio.ToCanonical(),  
            # tio.RescaleIntensity(out_min_max=(0, 1)),  
            # tio.CropOrPad((128, 128, 128)), 
        ])
    }
    train_dataset = MyDataSet(data_path_train, train_txt_path, transform=data_transform["train"])
    
    val_dataset = MyDataSet(data_path_val, val_txt_path, transform=data_transform["val"])



    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))




    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)

    # create model
   
    model =  Transformer3D().to(device)


    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=args.mm,weight_decay=args.wd)
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999))
    

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_auc = 0
    iter_num = 0
    name = args.name

    # save parameters path
    save_path = os.path.join(os.getcwd(), 'results/weights')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    train_data_loader = tqdm(train_loader, file=sys.stdout)
    for epoch in range(args.epochs):
        # train
        iter_num = 0
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                               device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup)


        scheduler.step()


        threshold, train_loss, train_auc = evaluate(model, train_data_loader, device)
        
        with open(os.path.join(save_path, "Net_log_train.txt"), 'a') as f:
            f.writelines('[epoch %d] train_loss: %.3f  train_auc: %.3f  ' % (
            epoch + 1, mean_loss, train_auc) + '\n')
        
        
       
        print(
            "[train iteration {}] loss: {:.3f}, auc: {:.3f} ".format(
                iter_num, train_loss, train_auc, ))
        iter_num = iter_num + 1
 
    # validation
    threshold, train_loss, train_auc = evaluate(model, train_data_loader, device)
    val_data_loader = tqdm(val_loader)

    threshold,  val_loss, val_auc= evaluate1(
        model, val_data_loader, device, threshold=threshold)

    with open(os.path.join(save_path, "Net_log_val.txt"), 'a') as f:
        f.writelines('[epoch %d] train_loss: %.3f  train_auc: %.3f  val_auc: %.3f' % (
        epoch + 1, mean_loss, train_auc, val_auc) + '\n')


      
    torch.save(model.state_dict(),
                           "/root/" + name + ".pth")


main(opt)
