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


def infer(args):
    # 随机种子, 保证结果可复现性
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_train = get_dataloader(args, 'train', seed=SEED)
    dataloader_val = get_dataloader(args, 'val', seed=SEED)
    dataloader_test = get_dataloader(args, 'test', seed=SEED)
    args.total_iter = args.total_epoch * len(dataloader_train)

    if args.device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    model = get_net(args)
    optimizer = utils.get_optimizer(args, model)
    lr_scheduler = utils.get_lr_scheduler(args, optimizer)
    loss_fn = utils.get_loss_fn(args.loss_fn)
    best_path = Path(args.checkpoint_root) / args.best_model_name
    logger = Logger(name='logger', log_path=Path(args.save_path) / 'log.txt')
    logger.write_dict(args.__dict__)
    logger.write_params_flops(model)

    state_dict = torch.load(best_path, weights_only=False)
    print("=> loaded best_model: epoch {}".format(state_dict['epoch']))
    model.load_state_dict(state_dict['model_dict'])
    # 计算 fps
    # utils.test_fps(model)
    val(args, dataloader_test, model, loss_fn, optimizer.param_groups[0]["lr"], -1, logger)


@torch.no_grad()
def val(args, dataloader, model, loss_fn, lr, epoch, logger):
    model.eval()
    cm = ConfusionMatrix(n_class=2)
    mode = 'test' if epoch == -1 else 'valid'
    print(f"\n================== begin to {mode}!!! ==================")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f'Valid Epoch {epoch}' if mode == 'valid' else 'Testing')

    epoch_loss = []
    img_list = []
    for idx, batch in pbar:
        pre_imgs = batch['pre_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        post_imgs = batch['post_imgs'].to(args.device, dtype=torch.float32, non_blocking=True)
        labels = batch['labels'].to(args.device, dtype=torch.long)

        output = model((pre_imgs, post_imgs))
        loss = loss_fn(output[0], labels)
        for i in range(1, len(output)):
            # mask1, 2, 3, 4 做深度监督
            aux_loss = loss_fn(output[i], labels)
            loss += aux_loss

        epoch_loss.append(loss.data.item())
        pred = torch.argmax(output[0], dim=1, keepdim=True)
        f1 = cm.update_cm(pr=pred.cpu().numpy(), gt=labels.cpu().numpy())
        pbar.set_postfix({
            'f1': f'{f1 * 100: .2f}',
            'loss': f'{loss.item():.4f}',
        })
        # if mode == 'test':
        #     img_list.append({'f1': f1,
        #                      'img_name': batch['img_name'][0],
        #                      'pred': pred,
        #                      'label': labels
        #                      })
        del pre_imgs, post_imgs, output, loss, labels
    # utils.save_imgs(args.vis_root, img_list)
    avg_loss = round(sum(epoch_loss) / len(epoch_loss), 6)
    scores_dict = cm.get_scores()
    text = f"epoch={epoch}, f1_0={scores_dict['f1'][0]:<.6f}, f1_1={scores_dict['f1'][1]:<.6f}, " \
           f"recall_0={scores_dict['recall'][0]:<.6f}, recall_1={scores_dict['recall'][1]:<.6f}, " \
           f"precision_0={scores_dict['precision'][0]:<.6f}, precision_1={scores_dict['precision'][1]:<.6f}, " \
           f"iou_0={scores_dict['iou'][0]:<.6f}, iou_1={scores_dict['iou'][1]:<.6f}, oa={scores_dict['oa']:<.6f}, " \
           f"kappa={scores_dict['kappa']:<.6f}, loss={avg_loss:<.6f}, lr={lr:<.6f}\n"
    logger.write(text)
    return round(scores_dict['f1'][1], 6), avg_loss
