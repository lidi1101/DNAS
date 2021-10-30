import logging
import time
import datetime

import torch

from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import reduce_loss_dict, compute_params
from .inference import derain_inference, denoise_inference, denoise_SIM_noise1800_inference
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR
from decompositions.decompositions import cp_decomposition_conv_layer,tucker_decomposition_conv_layer


def do_train(
        model,
        train_loader,
        val_list,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfg):
    """
    num_classes (int): number of classes. Required by computing mIoU.
    """
    logger = logging.getLogger("one_stage_nas.trainer")
    logger.info("Model:")
    logger.info(model)
    logger.info("Model Params: {:.2f}M".format(compute_params(model) / 1024 / 1024))
    logger.info("Model Params: {}".format(compute_params(model)))


    ####################################################################################################################################
    # lidi modify
    # 2020-07-14

    # print("----------------------------------------------------------------------------------------------")
    #
    # model.cpu()
    #
    # for layer in model.named_modules():
    #     if isinstance(layer[1],torch.nn.modules.conv.Conv2d):
    #         conv_layer=layer
    #         # rank=max(conv_layer.weight.data.numpy().shape)//3
    #         # decompose = cp_decomposition_conv_layer(conv_layer,rank)
    #         decompose=tucker_decomposition_conv_layer(conv_layer[1],[16,16])
    #         layer=decompose
    #
    # print(model)
    # model.cuda()

    ######################################################################################################################################

    logger.info("Start training")

    start_iter = arguments["iteration"]
    start_training_time = time.time()

    if cfg.DATASET.TASK == 'derain':
        inference = derain_inference
    elif cfg.DATASET.TASK == 'denoise':
        if cfg.DATASET.DATA_NAME in ['SIM_noise1800_train', 'SIM_noise1800_test']:
            inference = denoise_SIM_noise1800_inference
        else:
            inference = denoise_inference

    best_val = 0
    model.train()
    data_iter = iter(train_loader)

    meters = MetricLogger(delimiter="  ")
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    end = time.time()
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        data_time = time.time() - end

        pred, loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()).mean()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        pred[pred>1.0] = 1.0
        pred[pred<0.0] = 0.0

        targets = targets.cuda()

        metric_SSIM(pred.detach(), targets, transpose=False)
        metric_PSNR(pred.detach(), targets)

        if iteration % (val_period // 4) == 0:
            logger.info(
                meters.delimiter.join(
                ["eta: {eta}",
                 "iter: {iter}",
                 "{meters}",
                 "lr: {lr:.6f}",
                 "max_mem: {memory:.0f}"]).format(
                     eta=eta_string,
                     iter=iteration,
                     meters=str(meters),
                     lr=optimizer.param_groups[0]['lr'],
                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

        if iteration % val_period == 0:
            train_ssim, train_psnr = metric_SSIM.metric_get(), metric_PSNR.metric_get()
            metric_SSIM.reset()
            metric_PSNR.reset()

            ssim, psnr, input_img, output_img, target_img = inference(model, val_list, cfg, show_img=True, tag='train')
            if best_val < (ssim + psnr/100):
                best_val = (ssim + psnr/100)
                checkpointer.save("model_best", **arguments)
            # set mode back to train
            model.train()
            writer.add_image('img/train/input', input_img, iteration)
            writer.add_image('img/train/output', output_img, iteration)
            writer.add_image('img/train/target', target_img, iteration)
            writer.add_scalars('SSIM', {'train_ssim': train_ssim, 'val_ssim': ssim}, iteration)
            writer.add_scalars('PSNR', {'train_psnr': train_psnr, 'val_psnr': psnr}, iteration)

        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))

    writer.close()

