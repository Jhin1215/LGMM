import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim.lr_scheduler
from tqdm import tqdm

from models.model import get_net
from tools.dataloader import get_dataloader
from tools import utils
from tools.timer import Timer
from tools.logger import Logger
from tools.metric_tool import ConfusionMatrix


def fit(args):
    # 随机种子, 保证结果可复现性
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_train = get_dataloader(args, 'train', seed=SEED)
    dataloader_val = get_dataloader(args, 'val', seed=SEED)
    dataloader_test = get_dataloader(args, 'test', seed=SEED)
    args.total_iter = args.total_epoch * len(dataloader_train)

    if args.device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    model = get_net(args).to(args.device)

    optimizer = utils.get_optimizer(args, model)
    lr_scheduler = utils.get_lr_scheduler(args, optimizer)
    loss_fn = utils.get_loss_fn(args.loss_fn)

    # 训练过程变量
    cur_iter = 0
    best_f1_val = 0
    best_f1_loss = 1e3
    args.start_epoch = 0
    resume_path = Path(args.checkpoint_root) / 'last_model.pth'
    best_path = Path(args.checkpoint_root) / 'best_model.pth'

    if args.resume is not None and args.resume:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=args.device)
            args.start_epoch = checkpoint['epoch'] + 1
            cur_iter = args.start_epoch * len(dataloader_train)
            model.load_state_dict(checkpoint['model_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

        if os.path.isfile(best_path):
            checkpoint = torch.load(best_path, map_location=args.device)
            best_f1_val = checkpoint['f1']
            best_f1_loss = checkpoint['loss']
        else:
            print("=> no checkpoint found at '{}'".format(best_path))

    timer = Timer(args.total_iter)
    logger = Logger(name='logger', log_path=Path(args.save_path) / 'log.txt')
    logger.write_dict(args.__dict__)
    logger.write_params_flops(model)

    # ===== 训练 + 验证 =====
    for epoch in range(args.start_epoch, args.total_epoch):
        _ = train_without_amp(args, dataloader_train, model, optimizer, loss_fn,
                              epoch, cur_iter, logger, timer, lr_scheduler)
        cur_iter += len(dataloader_train)

        f1_val, f1_loss = val(args, dataloader_val, model, loss_fn,
                              optimizer.param_groups[0]["lr"], epoch, logger)

        # 保存最近一次断点
        torch.save({
            'epoch': epoch,
            'model_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'lr_scheduler_dict': lr_scheduler.state_dict(),
            'f1': f1_val,
        }, resume_path)

        # 根据 f1 与 loss 保存最好模型
        if utils.judge_is_exist(epoch, f1_val, f1_loss, best_f1_val, best_f1_loss, args.dataset_name):
            best_f1_loss = f1_loss
            best_f1_val = f1_val
            torch.save({
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'f1': f1_val,
                'loss': f1_loss,
            }, best_path)
            print("===========> the best model has updated <===========\n")

    state_dict = torch.load(best_path, map_location=args.device)
    print("=> loaded best_model: epoch {}".format(state_dict['epoch']))
    model.load_state_dict(state_dict['model_dict'])
    _ = val(args, dataloader_test, model, loss_fn, optimizer.param_groups[0]["lr"], -1, logger)


def train_without_amp(args, dataloader, model, optimizer, loss_fn, epoch, cur_iter,
                      logger, timer, lr_scheduler):
    model.train()
    print(f"\n================== begin to training!!! ==================")
    cm = ConfusionMatrix(n_class=2)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train Epoch {epoch}')

    for idx, batch in pbar:
        pre_imgs = batch['pre_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        post_imgs = batch['post_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        labels = batch['labels'].to(args.device, dtype=torch.long, non_blocking=True)

        output = model((pre_imgs, post_imgs))  # DataParallel 会自动 scatter / gather
        del pre_imgs, post_imgs
        loss = loss_fn(output[0], labels)
        for i in range(1, len(output)):
            aux_loss = loss_fn(output[i], labels)
            loss += aux_loss

        with torch.no_grad():
            pred = torch.argmax(output[0], dim=1, keepdim=True)
            f1 = cm.update_cm(pr=pred.cpu().numpy(), gt=labels.cpu().numpy())
            pbar.set_postfix({
                'f1': f'{f1 * 100:.2f}',
                'lr': f'{optimizer.param_groups[0]["lr"]: .8f}',
                'loss': f'{loss.item():.4f}',
            })

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        timer.update()
        if idx % 100 == 0:
            remain_time = timer.get_remaining(cur_iter + idx)
            print(f'training mode = train, epoch={epoch}, idx={idx}, '
                  f'loss = {loss.item():.6f}, f1 = {f1:.4f}, remain_time = {remain_time}\n')
        del output, loss, labels

    scores_dict = cm.get_scores()
    if epoch % 30 == 0:
        torch.cuda.empty_cache()
    return scores_dict


@torch.no_grad()
def val(args, dataloader, model, loss_fn, lr, epoch, logger):
    model.eval()
    cm = ConfusionMatrix(n_class=2)
    mode = 'test' if epoch == -1 else 'valid'
    print(f"\n================== begin to {mode}!!! ==================")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f'Valid Epoch {epoch}' if mode == 'valid' else 'Testing')
      
    epoch_loss = []
    for idx, batch in pbar:
        pre_imgs = batch['pre_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        post_imgs = batch['post_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        labels = batch['labels'].to(args.device, dtype=torch.long)

        output = model((pre_imgs, post_imgs))
        loss = loss_fn(output[0], labels)
        for i in range(1, len(output)):
            aux_loss = loss_fn(output[i], labels)
            loss += aux_loss

        # 指标：与你原有保持一致
        pred = torch.argmax(output[0], dim=1, keepdim=True)
        f1 = cm.update_cm(pr=pred.cpu().numpy(), gt=labels.cpu().numpy())

        epoch_loss.append(loss.item())
        pbar.set_postfix({'f1': f'{f1 * 100: .2f}', 'loss': f'{loss.item():.4f}'})

        del pre_imgs, post_imgs, output, loss, labels

    avg_loss = round(sum(epoch_loss) / len(epoch_loss), 6)
    scores_dict = cm.get_scores()
    text = (f"epoch={epoch}, f1_0={scores_dict['f1'][0]:<.6f}, f1_1={scores_dict['f1'][1]:<.6f}, "
            f"recall_0={scores_dict['recall'][0]:<.6f}, recall_1={scores_dict['recall'][1]:<.6f}, "
            f"precision_0={scores_dict['precision'][0]:<.6f}, precision_1={scores_dict['precision'][1]:<.6f}, "
            f"iou_0={scores_dict['iou'][0]:<.6f}, iou_1={scores_dict['iou'][1]:<.6f}, oa={scores_dict['oa']:<.6f}, "
            f"kappa={scores_dict['kappa']:<.6f}, loss={avg_loss:<.6f}, lr={lr:<.6f}\n")
    logger.write(text)
    return round(scores_dict['f1'][1], 6), avg_loss
    