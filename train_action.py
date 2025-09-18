
# 在代码开头添加环境变量设置（放在所有import之前）
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'  # 减少内存碎片
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更准确的错误定位


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更准确的错误定位

import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
torch.cuda.empty_cache()  # 清理缓存
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from lib.model.imbalanceCont_loss import *


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=110)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                # Ensure batch_gt is LongTensor
                batch_gt = batch_gt.long()
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            feature, output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)

            # 仅计算指标
            #CAC_score = class_alignment_consistency(feature, batch_gt, args.cac_alpha)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc1 = accuracy(output, batch_gt, topk=(1, 1))
            top1.update(acc1[0], batch_size)
            #top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                #print('Test CAC'+CAC_score)

    return losses.avg, top1.avg


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    #criterion = torch.nn.CrossEntropyLoss()
    # 修改损失函数：标签平滑，效果：缓解过拟合，提升模型校准性
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # PyTorch 1.10+

    # 初始化
    criterion_contra = SupConProtoLoss(
        num_classes=args.action_classes,  # 四分类：Angry, Neutral, Happy, Sad
        feature_dim=512,  # 你的 embedding 维度
        temperature_major=args.temperature_major,
        temperature_minor=args.temperature_minor
    )
    criterion_contra.to("cuda")

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda() 
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    '''
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    '''
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
    }
    '''
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    '''
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)

    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        # 混合精度训练和梯度裁剪
        scaler = GradScaler()
        accumulation_steps = 4  # 梯度累积步数
        clip_grad_norm = 1.0  # 梯度裁剪阈值
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            grad_norm = AverageMeter()  # 监控梯度范数
            model.train()
            end = time.time()

            # 学习率预热
            if epoch < getattr(args, 'warmup_epochs', 8):
                lr_scale = (epoch + 1) / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * lr_scale

            #iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 16, 4)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)

                if torch.cuda.is_available():
                    batch_gt = batch_gt.long().cuda()# Ensure the labels are of type torch.long
                    batch_input = batch_input.cuda()

                # Mixed precision context manager for forward pass
                with autocast():
                    # 前向
                    feature, output = model(batch_input)  # feature: [N, D], output: [N, num_classes]

                    # ----------------------
                    # 交叉熵损失
                    # ----------------------
                    batch_gt = batch_gt.squeeze().long()  # 确保 shape [N]
                    loss_crossent = criterion(output, batch_gt)

                    # ----------------------
                    # SupCon + Prototype 对比损失
                    # ----------------------
                    # 确保 feature 归一化
                    feature = F.normalize(feature, dim=1)
                    loss_contra = criterion_contra(feature, batch_gt)

                    # ----------------------
                    # 总损失
                    # ----------------------
                    loss_train = loss_crossent + args.alpha * loss_contra

                    loss_train = loss_train / accumulation_steps  # 梯度累积
                # Scales the loss, performs backward pass, and updates optimizer
                scaler.scale(loss_train).backward()

                # Gradient accumulation step
                if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    # 梯度裁剪
                    scaler.unscale_(optimizer)   # 解除梯度缩放以便裁剪
                    current_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad_norm)
                    grad_norm.update(current_grad_norm)

                    # Update optimizer after accumulating gradients for 'accumulation_steps' steps
                    scaler.step(optimizer)
                    scaler.update()  # Updates the scale for next iteration
                    optimizer.zero_grad()

                losses_train.update(loss_train.item() * accumulation_steps, batch_size)
                acc1, _= accuracy(output, batch_gt, topk=(1, 1))
                top1.update(acc1[0], batch_size)

                torch.cuda.empty_cache()
                batch_time.update(time.time() - end)
                end = time.time()
                if (idx + 1) % opts.print_freq == 0:
                #if (idx + 1) % 8 == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses_train, top1=top1))
                    sys.stdout.flush()

            # 验证阶段
            test_loss, test_top1 = validate(test_loader, model, criterion)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f'[{timestamp}] Epoch: {epoch} | Test Loss: {test_loss:.4f}, Acc: {test_top1:.2f}%')

            # 记录TensorBoard
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('grad_norm', grad_norm.avg, epoch + 1)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
            train_writer.add_scalars('loss_compare',
                               {'train': losses_train.avg, 'test': test_loss}, epoch)
            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                print('acc1:test_top1')
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1= validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'.format(loss=test_loss, top1=test_top1))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)