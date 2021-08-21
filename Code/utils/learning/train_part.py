import shutil
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import time

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.hinet import HINet
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms

def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = 0
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output, aux_output = model(input)
        loss_hinet = loss_type(output, target, maximum) + loss_type(aux_output, target, maximum)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss_hinet.backward()
        optimizer.step()
        
        total_loss += loss.item()
        len_loader += 1

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
                'lr = ', optimizer.param_groups[0]['lr']
            )
        # if iter % args.tensorboard_interval == 0:
        #     tot_iter = iter + epoch * len(data_loader)
        #     writer.add_scalar('Loss/train', loss.item(), tot_iter)
        #     writer.add_scalar('Epoch/train', epoch, tot_iter)
        #     writer.add_scalar('lr/train', optimizer.param_groups[0]['lr'], tot_iter)

        start_iter = time.perf_counter()
    #scheduler.step()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, scheduler):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output, _ = model(input)

            for i in range(output.cpu().numpy().shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    print('val_num_subjects: ', num_subjects)
    scheduler.step(metric_loss/num_subjects)
    return metric_loss, num_subjects, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
    # if (epoch <= 15) or (epoch<=25 and epoch % 3 == 0) or (epoch <= 100 and epoch %5 == 0):
    #     shutil.copyfile(exp_dir / 'model.pt', exp_dir / (str(epoch) + '_model.pt'))

def lr_function(epoch):
    if epoch < 7:
        return 1
    elif epoch < 25:
        return 0.5
    else:
        return 0.25 * (0.985**epoch)
        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    # Class로 관리 요망
    #writer = SummaryWriter(log_dir = args.exp_dir, flush_secs=10, comment = args.net_name)

    model = HINet(in_chn = args.in_chans)
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_function, last_epoch = -1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.67, patience=1, threshold=0.00008, threshold_mode='abs', cooldown=0, min_lr=0, eps=0, verbose=False)


    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type)
        val_loss, num_subjects, val_time = validate(args, model, val_loader, scheduler)


        val_loss /= num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        #writer.add_scalar('val_loss/val', val_loss, epoch)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
