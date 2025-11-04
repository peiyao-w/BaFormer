import os
import argparse
import pathlib
import time
import random
# import matplotlib.colors as mcolors

import math
import numpy as np
# import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
import matplotlib.pyplot as plt
from PIL import Image
# import shutil



from action_segmentation import(
    augment_crop,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
    SetCriterion_bd,
)
from action_segmentation.config.config_node import ConfigNode
from action_segmentation.utils import (
    AverageMeter,
    AverageMeter_acc,
    AverageMeter_f1,
    compute_dense_acc,
    compute_f1,
    edit_score,
    DummyWriter,
    compute_metrics,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)



global_step = 0

def load_config():
    parser = argparse.ArgumentParser(description = "BaFormer for efficient temporal action segmentation.")
    parser.add_argument('--config', default="configs/framed_en_de.yaml", type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()

    if args.config is not None:
        config.merge_from_file(args.config, allow_unsafe=True)
    config.merge_from_list(args.options)

    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config = update_config(config)
    config.freeze()
    return config

def get_set_label(frame_label, num_cls):
    device = frame_label.device
    frame_label = frame_label.tolist()[0]
    frame_label.append(num_cls)
    frame_label = torch.tensor(frame_label)

    exist_set = torch.bincount(frame_label)[:-1].bool().int().to(device)
    return exist_set


def get_loss(input, target):# binary cross-entropy
    input = input.squeeze(0)
    target = target.float()
    assert input.shape == target.shape and len(input.shape) ==1
    act_func = nn.Sigmoid()
    loss_func = nn.BCELoss()
    loss = loss_func(act_func(input), target)
    return loss

class Meter_dict():
    def __init__(self,):
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter_acc()
        self.edit_meter = AverageMeter()
        self.f1_meter = AverageMeter_f1()

        self.ins_acc_meter = AverageMeter_acc()
        self.ord_acc_meter = AverageMeter_acc()
        self.cat_acc_meter = AverageMeter_acc()


    def get_update_metric(self,  dataset, seg_pred, frame_target, loss, ins_seg_pred): # for a batch, to record the metric value
        L = frame_target.shape[1]
        frame_target = frame_target.view(-1)

        seg_pred = seg_pred.view(-1, seg_pred.shape[-1])  # if use 50salads, it need interpeave
        assert seg_pred.shape[0] == frame_target.shape[0]
        num_correct, acc, edit, f_scores = compute_metrics(dataset, seg_pred, frame_target)

        ##-----ins seg
        ins_seg_pred = ins_seg_pred.view(-1, ins_seg_pred.shape[-1])  # if use 50salads, it need interpeave
        assert ins_seg_pred.shape[0] == frame_target.shape[0]
        ins_num_correct, ins_acc, ins_edit, ins_f_scores = compute_metrics(dataset, ins_seg_pred, frame_target)


        loss = loss.item()
        num_correct = num_correct.item()
        acc = acc.item()

        num = frame_target.shape[0]
        self.loss_meter.update(loss, num)
        self.acc_meter.update(num_correct, acc, num)
        self.edit_meter.update(edit, 1)
        self.f1_meter.update(f_scores)

        self.ins_acc_meter.update(ins_num_correct, ins_acc, num)


        metric_results = { 'loss_meter': self.loss_meter,
                            'acc_meter': self.acc_meter,
                            'edit_meter': self.edit_meter,
                            'f1_meter': self.f1_meter,
                           'ins_acc_meter': self.ins_acc_meter,
                           }
        return metric_results

class Record_dict():
    def __init__(self):  #
        self.reset()
    def reset(self):  # the params needed to update initialize here!
        self.acc_max = 0
        self.edit_max = 0
        self.f1_max = (0, 0, 0)
        self.best_epoch = 0
    def update_record(self, epoch, acc, edit, f1):
        if_update = False
        if epoch == 0:
            self.reset()
        else:
            if acc > self.acc_max:
                self.acc_max = acc
                self.edit_max = edit
                self.f1_max = f1
                self.best_epoch = epoch
                if_update = True
        return if_update

def create_heatmap(sequence, sigma=2.0):
    device = sequence.device
    # Find the positions of '1' in the sequence
    sigma = torch.tensor([sigma]).to(device)
    peak_positions = torch.nonzero(sequence).squeeze(dim=-1)

    # Create a 1D tensor representing the indices
    indices = torch.arange(0, len(sequence)).to(device)

    # Calculate Gaussian functions for each peak position
    gaussians = [torch.exp(-(indices - peak_pos).float() ** 2 / (2 * sigma ** 2)) for peak_pos in peak_positions]

    # Sum up the Gaussian functions
    heatmap = torch.max(torch.stack(gaussians), dim=0)[0]

    return heatmap

def prepare_target(frame_tgs, sigma):
    b, l = frame_tgs.shape #just 1 sample, b=1
    frame_tg = frame_tgs.squeeze(dim=0)
    gt_mask = []
    gt_class = []
    current = frame_tg[0]
    start = 0
    frame_bd = torch.zeros_like(frame_tg)
    for j in range(l):
        if j != l-1:
            if frame_tg[j] != current:
                end = j
                mask = torch.zeros(l).to(frame_tgs)
                mask[start: end] = 1
                if sigma is not None:
                    mask = create_heatmap(mask, sigma)
                # # Plotting the binary sequence and the heatmap
                # plt.subplot(2, 1, 1)
                # plt.title('Binary Sequence')
                # plt.stem(mask.cpu().numpy(), linefmt='b-', markerfmt='bo', basefmt='k-')
                # plt.subplot(2, 1, 2)
                # plt.title('Heatmap')
                # plt.plot(mask1.cpu().numpy(), 'r-')
                # plt.tight_layout()
                # plt.show()

                gt_mask.append(mask.unsqueeze(dim=0))
                gt_class.append(current.unsqueeze(dim=0))
                frame_bd[end] = 1 #start of the next segment
                start = j
                current = frame_tg[j]

        else: #the final frame
            end = l # j=l-1
            mask = torch.zeros(l).to(frame_tgs)
            mask[start: end] = 1
            if sigma is not None:
                mask = create_heatmap(mask, sigma)
            gt_mask.append(mask.unsqueeze(dim=0))
            gt_class.append(current.unsqueeze(dim=0))
    gt_mask = torch.cat(gt_mask)
    gt_class = torch.cat(gt_class)
    target = {  'labels': gt_class,
                'masks': gt_mask,
                'boundarys': frame_bd}
    targets = [target] # if we have 1 sample in a batch
    return targets


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum('qc,ql->cl', mask_cls, mask_pred).transpose(0,1)
    return semseg
def inference(prediction):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    processed_results = []
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
        r = semantic_inference(mask_cls_result, mask_pred_result)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred


def inference_bd(prediction):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_ref = torch.zeros_like(mask_pred)
        mask_bd = (mask_bd.sigmoid()>0.3).float().squeeze(dim=0)
        mask_bd[0] = 1
        mask_bd[-1] = 1
        indices = torch.nonzero(mask_bd)
        for i in range(len(indices)-1):
            star, end = indices[i], indices[i+1]
            mask_split = mask_pred[:, star:end].sigmoid().sum(dim=-1).argmax(0)
            mask_ref[mask_split,star:end] = 1
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls, mask_ref).transpose(0, 1)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred

def inference_bd_peak(prediction, threshold):
    assert 'pred_logits' in prediction
    assert 'pred_masks' in prediction
    assert 'pred_boundarys' in prediction
    mask_cls_results = prediction['pred_logits']
    mask_pred_results = prediction['pred_masks']
    mask_boundary_results = prediction['pred_boundarys']
    processed_results = []
    for mask_cls, mask_pred, mask_bd in zip(mask_cls_results, mask_pred_results, mask_boundary_results):
        mask_bd = mask_bd.sigmoid().float().squeeze(dim=0)
        mask_bd[mask_bd < threshold] = 0.0
        peak = torch.where((mask_bd[ :-2] < mask_bd[1:-1])
                        & (mask_bd[ 2:] < mask_bd[1:-1] ))[0]
        indices = [0] + peak.tolist() + [mask_bd.shape[-1]]
        mask_ref = torch.zeros_like(mask_pred)
        for i in range(len(indices)-1):
            star, end = indices[i], indices[i+1]
            mask_split = mask_pred[:, star:end].sigmoid().sum(dim=-1).argmax(0)
            mask_ref[mask_split,star:end] = 1
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        r = torch.einsum('qc,ql->cl', mask_cls, mask_ref).transpose(0, 1)
        r = _relabeling(r, 100)
        processed_results.append(r)
    seg_pred = torch.stack(processed_results, dim=0)
    return seg_pred

def _relabeling(outputs, theta_t):
    preds = outputs.argmax(dim=1)
    last = preds[0]
    cnt = 1
    for j in range(1, len(preds)):
        if last == preds[j]:
            cnt += 1
        else:
            if cnt > theta_t:
                cnt = 1
                last = preds[j]
            else:
                outputs[j - cnt : j, :] = outputs[j - cnt - 1, :]
                cnt = 1
                last = preds[j]

    if cnt <= theta_t:
        outputs[j - cnt: j, :] = outputs[j - cnt - 1, :]

    return outputs


# train one epoch
def train(epoch, config, model, optimizer, scheduler, train_loader, logger, tensorboard_writer, tensorboard_writer2):
    global global_step

    logger.info(f'Train {epoch} {global_step}')
    device = torch.device(config.device)
    model.train()

    #-----init meter_dict
    meter_dict = Meter_dict()
    new_edit_meter = AverageMeter()
    before_edit_meter = AverageMeter()

    criterion = SetCriterion_bd(config)

    step = 0

    for idx_sample, (data, frame_target, fname, noise) in enumerate(train_loader):# data(1, 2048, l) , target (1,l)
        idx_sample += 1
        if config.augmentation.is_use:
            data, target = augment_crop(data, frame_target)

        if config.dataset.noise_weight is not None:
            data = data + config.dataset.noise_weight * noise.type(torch.float32)
        data = data.to(device, non_blocking=config.train.dataloader.non_blocking)
        frame_target = frame_target.to(device, non_blocking=config.train.dataloader.non_blocking).long() #[1, L] frame-wise
        targets = prepare_target(frame_target, config.dataset.guassian_sigma)
        # print(targets[0]['labels'])

        #------- change target to set_label for multi-calssification
        optimizer.zero_grad()


        outputs = model(data)

        ##-----multi-query: instance, ordinal, categorical
        weight_dict = {"loss_ce": config.model.ce_weight,
                       "loss_mask": config.model.mask_weight,
                       "loss_dice": config.model.dice_weight,
                       "loss_bd": config.model.bd_weight,}
        if config.model.action_seg.transformer_decoder.deep_supervision:
            dec_layers = config.model.action_seg.transformer_decoder.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if config.model.pose_weight_single != 0:
            pos_weight_info = (config.dataset.name, config.dataset.dataset_dir, fname, config.model.pose_weight_single)
        else:
            pos_weight_info = None

        losses = criterion(outputs, targets, pos_weight_info)
        for k in list(losses.keys()):
            if k in weight_dict:
                losses[k] *= weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        #from detectron2
        loss_dict = losses
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        loss = losses
        # print(loss_dict)

        loss.backward()
        optimizer.step()

        # ------inference
        # ins_seg_pred = inference(outputs)
        ins_seg_pred = inference_bd_peak(outputs, config.dataset.threshold)
        seg_pred = ins_seg_pred.softmax(dim=-1) # + ord_seg_pred.softmax(dim=-1) + cat_seg_pred.softmax(dim=-1)

        #------compute metrics
        metric_results = meter_dict.get_update_metric(config.dataset.name, seg_pred, frame_target, loss, ins_seg_pred) #loss_meter and acc_meter

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        idx_batch_sample = 0
        step +=1
        global_step += 1
        if step % config.train.log_period == 0 or idx_sample == len(train_loader):  # per batch and the final one, to print
            loss_meter = metric_results['loss_meter']
            acc_meter = metric_results['acc_meter']
            edit_meter = metric_results['edit_meter']
            f1_meter = metric_results['f1_meter']
            ins_acc_meter = metric_results['ins_acc_meter']

            total = math.ceil(len(train_loader) / config.train.batch_size)
            logger.info(
                f'Epoch {epoch} '
                f'Step {step}/{total} '  # in epoch
                f'lr {scheduler.get_last_lr()[0]:.6f} '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'acc% {acc_meter.val*100:.4f} ({acc_meter.avg*100:.4f}) '
                f'ins_acc% {ins_acc_meter.val * 100:.4f} ({ins_acc_meter.avg * 100:.4f}) '
                f'edit {edit_meter.val:.4f}({edit_meter.avg:.4f}) '
                f'f1@10 {f1_meter.val[0]:.4f}({f1_meter.avg[0]:.4f}) '
                f'f1@25 {f1_meter.val[1]:.4f}({f1_meter.avg[1]:.4f}) '
                f'f1@50 {f1_meter.val[2]:.4f}({f1_meter.avg[2]:.4f}) '
                )

    scheduler.step() # according to the epoch

@torch.no_grad()
def validate(epoch, config, model,  val_loader, val_record, logger, tensorboard_writer):
    device = torch.device(config.device)
    meter_dict = Meter_dict()
    criterion = SetCriterion_bd(config)


    new_edit_meter = AverageMeter()
    before_edit_meter = AverageMeter()

    model.eval()

    for step, (data, frame_target, fname) in enumerate(val_loader): # batchsize=1, one by one
        #----- data process
        data = data.to(device, non_blocking=config.validation.dataloader.non_blocking)
        frame_target = frame_target.to(device, non_blocking=config.validation.dataloader.non_blocking).long()
        targets = prepare_target(frame_target, config.dataset.guassian_sigma)

        outputs = model(data)

        ##-------other sample rate
        if config.dataset.sample_rate != 1:
            L = frame_target.shape[-1]
            outputs['pred_masks'] = outputs['pred_masks'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
            # outputs['pred_boundarys'] = outputs['pred_boundarys'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
            outputs['pred_boundarys'] = F.interpolate(outputs['pred_boundarys'],size = L, mode='linear' )


            aux_outputs_resize = []
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    aux_outputs['pred_masks'] = aux_outputs['pred_masks'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
                    aux_outputs['pred_boundarys'] = aux_outputs['pred_boundarys'].repeat_interleave(config.dataset.sample_rate, dim = -1)[:, :, :L]
                    aux_outputs_resize.append({'pred_logits':aux_outputs['pred_logits'],
                                                'pred_masks':aux_outputs['pred_masks'],
                                                'pred_boundarys': aux_outputs['pred_boundarys'] })
            outputs['aux_outputs'] = aux_outputs_resize
        ##-----multi-query: instance, ordinal, categorical
        weight_dict = {"loss_ce": config.model.ce_weight,
                       "loss_mask": config.model.mask_weight,
                       "loss_dice": config.model.dice_weight}

        losses = criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in weight_dict:
                losses[k] *= weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        #from detectron2
        loss_dict = losses
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        loss = losses

        #------inference
        # ins_seg_pred = inference(outputs)
        ins_seg_pred = inference_bd_peak(outputs, config.dataset.threshold)
        seg_pred = ins_seg_pred.softmax(dim=-1)

        #------compute metrics
        metric_results = meter_dict.get_update_metric(config.dataset.name, seg_pred, frame_target, loss, ins_seg_pred) #loss_meter and acc_meter

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # the test don't separate to multiply batch, so just compute thr average results
    loss_avg = metric_results['loss_meter'].avg
    acc_avg = metric_results['acc_meter'].avg
    edit_avg = metric_results['edit_meter'].avg
    f1_avg = metric_results['f1_meter'].avg

    ins_acc_avg = metric_results['ins_acc_meter'].avg

    if_update = val_record.update_record(epoch, acc_avg, edit_avg, f1_avg) # update and return something

    acc_max = val_record.acc_max
    edit_max = val_record.edit_max
    f1_max = val_record.f1_max

    logger.info(
        f'----------------------------------------------------------------------------------------------------------------------------------------------------'
        f'Val {epoch:3d} '
        f'loss {loss_avg:.4f} | '
        f'acc% {acc_avg*100:.4f} | '
        f'ins_acc% {ins_acc_avg*100:.4f} | '
        f'edit {edit_avg:.4f} | '
        f'f1@10 {f1_avg[0]:.4f} | '
        f'f1@25 {f1_avg[1]:.4f} | '
        f'f1@50 {f1_avg[2]:.4f} |------|'
        f'acc_max% {acc_max*100:.4f} | '
        f'f1@1_max {f1_max[0]:.4f} | '
        f'f1@25_max {f1_max[1]:.4f} | '
        f'f1@50_max {f1_max[2]:.4f} |'
        f'edit_max {edit_max: 4f} | '        
        )


    # if epoch > 0:
    #     tensorboard_writer.add_scalar('Val/Loss', loss_avg, epoch)
    # tensorboard_writer.add_scalar('Val/FrameAcc', frame_acc_avg * 100, epoch)
    # tensorboard_writer.add_scalar('Val/SetAcc', set_acc_avg * 100, epoch)
    return if_update

def main():
    global global_step

    config = load_config()
    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)
    out_dir = os.path.join(config.train.output_dir, config.dataset.name, config.model.name, config.model.note, str(config.dataset.split))
    output_dir = pathlib.Path(out_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    if not config.train.resume:
        save_config(config, output_dir / 'config.yaml')
        save_config(get_env_info(config), output_dir / 'env.yaml')
        diff = find_config_diff(config)
        if diff is not None:
            save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)

    model = create_model(config)

    optimizer = create_optimizer(config, model)

    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch = len(train_loader))

    checkpointer = Checkpointer(model,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              save_dir=output_dir,
                              save_to_disk=get_rank() == 0)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()
    elif config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    if os.path.exists(output_dir / 'epoch_logs'):
        os.removedirs(output_dir / 'epoch_logs')
        print('Remove the epoch_logs successuflly!')
    if os.path.exists(output_dir / 'step_logs'):
        os.removedirs(output_dir / 'step_logs')
        print('Remove the step_logs successuflly!')

    if config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
            config, output_dir/'logs_epoch', purge_step=config.train.start_epoch +1)
        tensorboard_writer2 = create_tensorboard_writer(
            config, output_dir / 'logs_epoch'/'logs_step', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()

    # print('saving to:', out_dir)
    val_record = Record_dict()


    # if (config.train.val_period > 0 and start_epoch == 0 and config.train.val_first):
    #     if_update = validate(0, config, model, val_loader, val_record, logger, tensorboard_writer)

    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch +=1
        # np.random.seed(seed)
        train(epoch, config, model, optimizer, scheduler, train_loader, logger, tensorboard_writer, tensorboard_writer2)

        if config.train.val_period > 0 and (epoch % config.train.val_period == 0):
            if_update = validate(epoch, config, model, val_loader, val_record, logger, tensorboard_writer)

        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        if if_update:
            checkpoint_config = {
                'epoch': epoch,
                'global_step': global_step,
                'config': config.as_dict(),
            }
            checkpointer.save(f'checkpoint_best', **checkpoint_config)# save current epoch
    tensorboard_writer.close()
    tensorboard_writer2.close()

if __name__ =='__main__':
    main()