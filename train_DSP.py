import argparse
import os
import sys
import random
import timeit
import datetime
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab
import glob
from utils.misce import *
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate
import imageio
import time

start = timeit.default_timer()

start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='./configs/configUDA_gta.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default="GTA2City",
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()

def evaluate_wrapper(model, ignore_label=255, save_dir=None):
    if source_dataset_name == 'gta':
        return evaluate(model=model, dataset='cityscapes', ignore_label=ignore_label, input_size=(512,1024), save_dir=save_dir)
    elif source_dataset_name == 'synthia':
        return evaluate(model=model, dataset='cityscapes16', ignore_label=ignore_label, input_size=(512, 1024), save_dir=save_dir)

def main():
    print(config)

    best_mIoU = 0

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model, num_classes, gpus)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    cudnn.benchmark = True

    if random_crop:
        data_aug = Compose([RandomCrop_gta(input_size)])
    else:
        data_aug = None

    #New loader for Domain transfer
    if source_dataset_name == 'gta':
        source_loader = get_loader('gta')
        source_path = get_data_path('gta')

        #data_aug = Compose([RandomHorizontallyFlip()])
        source_dataset = source_loader(source_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug,
                                       img_size=(1280,720), mean=IMG_MEAN)
        sourceloader = data.DataLoader(source_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        sourceloader_iter = iter(sourceloader)

        target_loader = get_loader('cityscapes')
        target_path = get_data_path('cityscapes')
        # data_aug = Compose([RandomHorizontallyFlip()])
        target_dataset = target_loader(target_path, is_transform=True, augmentations=data_aug, img_size=input_size,
                                       img_mean=IMG_MEAN)
        np.random.seed(random_seed)
        targetloader = data.DataLoader(target_dataset,
                                       batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        targetloader_iter = iter(targetloader)

    #New loader for Domain transfer
    if source_dataset_name == 'synthia':
        source_loader = get_loader('synthia')
        source_path = get_data_path('synthia')
        source_dataset = source_loader(source_path, list_path='./data/synthia_list/train.txt', augmentations=data_aug,
                                       img_size=(1280, 760), mean=IMG_MEAN)
        sourceloader = data.DataLoader(source_dataset,
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        sourceloader_iter = iter(sourceloader)

        target_loader = get_loader('cityscapes16')
        target_path = get_data_path('cityscapes16')
        # data_aug = Compose([RandomHorizontallyFlip()])
        target_dataset = target_loader(target_path, is_transform=True, augmentations=data_aug, img_size=input_size,
                                       img_mean=IMG_MEAN)
        np.random.seed(random_seed)
        targetloader = data.DataLoader(target_dataset,
                                       batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        targetloader_iter = iter(targetloader)

    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    if source_dataset_name == 'gta':
        lt_cls_mixer = rand_mixer(get_data_path('gta'), "gta")
        class_to_select = [12, 15, 16, 17, 18]
    elif source_dataset_name == 'synthia':
        lt_cls_mixer = rand_mixer(get_data_path('synthia'), "synthia")
        class_to_select = [3, 4, 6, 13]

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_2_value = 0
        loss_mmd_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter, learning_rate, num_iterations, lr_power)

        # training loss for labeled data only
        try:
            batch = next(sourceloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourceloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourceloader_iter = iter(sourceloader)
            batch = next(sourceloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        weak_parameters={"flip": 0}

        images, labels, _, _ = batch

        images = images.cuda()
        labels = labels.cuda().long()

        #images, labels = weakTransform(weak_parameters, data = images, target = labels)

        pred = interp(model(images)[0])

        L_l = loss_calc(pred, labels, ignore_label, gpus) # Cross entropy loss for labeled data

        #L_l = torch.Tensor([0.0]).cuda()
        try:
            batch = next(sourceloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourceloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourceloader_iter = iter(sourceloader)
            batch = next(sourceloader_iter)

        image_mix, label_mix, _, _ = batch

        image_mix = image_mix.cuda()
        label_mix = label_mix.cuda().long()

        if train_unlabeled:
            try:
                batch_target = next(targetloader_iter)
                if batch_target[0].shape[0] != batch_size:
                    batch_target = next(targetloader_iter)
            except:
                targetloader_iter = iter(targetloader)
                batch_target = next(targetloader_iter)

            images_target, _, _, _, _ = batch_target
            images_target = images_target.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_target)
            #inputs_u_w = inputs_u_w.clone()
            logits_u_w = interp(ema_model(inputs_u_w)[0])
            logits_u_w, _ = weakTransform(weak_parameters, data = logits_u_w.detach())

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            cls_to_use = random.sample(class_to_select, 2)

            if mix_mask == "class":
                for image_i in range(batch_size):

                    classes = torch.unique(label_mix[image_i])

                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(label_mix[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(label_mix[image_i], classes).unsqueeze(0).cuda()

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            source_mix_img0, source_mix_lbl0, target_mix_img0, target_mix_lbl0, MixMask0_lam = \
                strongTransform_class_mix(image_mix[0], images[0], images_target[0], label_mix[0], labels[0], targets_u_w[0],
                                                                    MixMask0, lt_cls_mixer, cls_to_use, strong_parameters, mixWeight=lam)

            source_mix_img1, source_mix_lbl1, target_mix_img1, target_mix_lbl1, MixMask1_lam = \
                strongTransform_class_mix(image_mix[1], images[1], images_target[1], label_mix[1], labels[1], targets_u_w[1],
                                                                    MixMask0, lt_cls_mixer, cls_to_use, strong_parameters, mixWeight=lam)

            del image_mix, images

            target_mix_img = torch.cat((target_mix_img0, target_mix_img1))
            source_mix_img = torch.cat((source_mix_img0, source_mix_img1))
            tar_mix_p1, tar_mix_p2 = model(target_mix_img)
            tar_mix_pred = interp(tar_mix_p1) # target_mix_pred

            src_mix_p1, src_mix_p2 = model(source_mix_img)

            src_mix_pred = interp(src_mix_p1)  # source_mix_pred

            #p1,p2 = model(inputs_u_s)
            ap = nn.AdaptiveAvgPool2d((1,1))
            #logits_u_s = interp(p1)
            gs = ap(tar_mix_p2).flatten(1)
            p2 = interp(tar_mix_p2)

            f_source = MixMask0 * p2
            f_source = ap(f_source).flatten(1)

            #pt1,pt2 = model(inputs_t_s)
            #logits_t_s = interp(pt1)
            gt = ap(src_mix_p2).flatten(1)
            pt2 = interp(src_mix_p2)
            f_target = MixMask0 * pt2
            f_target = ap(f_target).flatten(1)


            loss_feature = mmd_rbf(f_source,f_target)
            loss_global = mmd_rbf(gs,gt)
            loss_mmd = 0.0001*loss_global + 0.0001*loss_feature

            target_mix_lbl = torch.cat((target_mix_lbl0, target_mix_lbl1)).long()
            source_mix_lbl = torch.cat((source_mix_lbl0, source_mix_lbl1)).long()

            L_l2 = unlabeled_loss(src_mix_pred, source_mix_lbl, torch.ones(source_mix_lbl.shape).cuda()*lam) \
                   + unlabeled_loss(src_mix_pred, labels, torch.ones(labels.shape).cuda()*(1-lam))

            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(target_mix_lbl.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().cuda()
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).cuda()

            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))

            pixelWiseWeight_a = (torch.cat((pixelWiseWeight0,pixelWiseWeight1))*lam).cuda()
            pixelWiseWeight_b = (torch.cat((pixelWiseWeight0,pixelWiseWeight1))*(1-lam)).cuda()

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(target_mix_lbl.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(tar_mix_pred, pseudo_label)
            elif consistency_loss == 'CE':

                L_u = consistency_weight * unlabeled_loss(tar_mix_pred, target_mix_lbl, pixelWiseWeight_a)
                + consistency_weight * unlabeled_loss(tar_mix_pred, targets_u_w, pixelWiseWeight_b)
            loss = L_l + L_u +  L_l2 + loss_mmd

        else:
            loss = L_l

        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            loss_2_value += L_l2.item()
            if train_unlabeled:
                loss_u_value += L_u.item()
                loss_mmd_value += loss_mmd.item()

        torch.cuda.empty_cache()
        loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter, gpus=gpus)
        if i_iter % 100 == 0 :

            print('iter = {0:6d}/{1:6d}, loss_l = {2:.4f}, loss_u = {3:.4f}, loss_2 = {4:.4f}, lambda = {5:.4f}, loss_mmd = {6:.4f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value, loss_2_value,lam,loss_mmd_value))

        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            mIoU = evaluate_wrapper(model,  ignore_label=255, save_dir=checkpoint_dir)

            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                save_checkpoint(mIoU, i_iter, model, optimizer, config, ema_model, checkpoint_dir, train_unlabeled, save_best=True, gpus=gpus)

            print('The best miou is %.4f' % best_mIoU)


    #_save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    mIoU = evaluate_wrapper(model,  ignore_label=255, save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        save_checkpoint(mIoU, i_iter, model, optimizer, config, ema_model, checkpoint_dir, train_unlabeled, save_best=True, gpus=gpus)

    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')
    args = get_arguments()
    config = json.load(open(args.config))

    model = config['model']

    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    source_dataset_name = config['training']['source_dataset']['name']
    num_classes = config['training']['source_dataset']['num_classes']

    lam = config['training']['lam']

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label']

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0, 1, 2, 3)[:args.gpus]

    main()
