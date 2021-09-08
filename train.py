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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_ammend(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_class_mix(image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters):
    inputs_, _ = transformsgpu.oneMix(mask_img, data=torch.cat((image1.unsqueeze(0), image2.unsqueeze(0))))
    _, targets_ = transformsgpu.oneMix(mask_lbl, target=torch.cat((label1.unsqueeze(0), label2.unsqueeze(0))))
    inputs_, targets_ = cls_mixer.mix(inputs_.squeeze(0), targets_.squeeze(0), cls_list)
    out_img, out_lbl = strongTransform_ammend(strong_parameters, data=inputs_, target=targets_)
    return out_img, out_lbl

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target



class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(miou,iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filelist = glob.glob(os.path.join(checkpoint_dir,'*.pth'))
        if filelist:
            os.remove(filelist[0])
        filename = os.path.join(checkpoint_dir, f'{miou}best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

class rand_mixer():
    def __init__(self, root, dataset):
        if dataset == "gta5":
            jpath = './data/gta5_ids2path.json'
            self.resize = (1280, 720)
            input_size = (512, 512)
            self.data_aug = Compose([RandomCrop_gta(input_size)])
        elif dataset == "cityscapes":
            jpath = './data/cityscapes_ids2path.json'
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        self.root = root
        self.dataset = dataset
        self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, classes):
        img_size = in_lbl.shape
        for i in classes:
            if self.dataset == "gta5":
                while(True):
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join(self.root, "images/%s" % name[0])
                    label_path = os.path.join(self.root, "labels/%s" % name[0])
                    img = Image.open(img_path)
                    lbl = Image.open(label_path)
                    img = img.resize(self.resize, Image.BICUBIC)
                    lbl = lbl.resize(self.resize, Image.NEAREST)
                    img = np.array(img, dtype=np.uint8)
                    lbl = np.array(lbl, dtype=np.uint8)
                    img, lbl = self.data_aug(img, lbl) # random crop to input_size
                    img = np.asarray(img, np.float32)
                    lbl = np.asarray(lbl, np.float32)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy:
                        lbl = label_copy.copy()
                        img = img[:, :, ::-1].copy()  # change to BGR
                        img -= IMG_MEAN
                        img = img.transpose((2, 0, 1))
                        break
                img = torch.Tensor(img).cuda()
                lbl = torch.Tensor(lbl).cuda()
                class_i = torch.Tensor([i]).type(torch.int64).cuda()
                MixMask = transformmasks.generate_class_mask(lbl, class_i)
                mixdata = torch.cat((img.unsqueeze(0), in_img.unsqueeze(0)))
                mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl.unsqueeze(0)))
                data, target = transformsgpu.oneMix(MixMask, data=mixdata, target=mixtarget)
                return data, target

def main():
    print(config)

    best_mIoU = 0

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()

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
        ema_model = create_ema_model(model)
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
    target_loader = get_loader('cityscapes')
    target_path = get_data_path('cityscapes')

    if random_crop:
        data_aug = Compose([RandomCrop_city(input_size)])
    else:
        data_aug = None

    #data_aug = Compose([RandomHorizontallyFlip()])
    target_dataset = target_loader(target_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)


    np.random.seed(random_seed)
    targetloader = data.DataLoader(target_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    targetloader_iter = iter(targetloader)


    #New loader for Domain transfer

    source_loader = get_loader('gta')
    source_path = get_data_path('gta')
    if random_crop:
        data_aug = Compose([RandomCrop_gta(input_size)])
    else:
        data_aug = None

    #data_aug = Compose([RandomHorizontallyFlip()])
    source_dataset = source_loader(source_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN)

    sourceloader = data.DataLoader(source_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    sourceloader_iter = iter(sourceloader)


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

    accumulated_loss_l = []
    accumulated_loss_u = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    gta5_cls_mixer = rand_mixer(get_data_path('gta'), "gta5")
    class_to_select = [12, 15, 16, 17, 18]

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_2_value = 0
        loss_mmd_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

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

        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data

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



        lam = 0.9
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

            if mix_mask == "class":
                for image_i in range(batch_size):

                    classes = torch.unique(label_mix[image_i])

                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(label_mix[image_i], classes).unsqueeze(0).cuda()
                        MixMask0_lam = MixMask0 * lam
                    else:
                        MixMask1 = transformmasks.generate_class_mask(label_mix[image_i], classes).unsqueeze(0).cuda()
                        MixMask1_lam = MixMask1 * lam

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))


            strong_parameters = {"Mix": MixMask0_lam}
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

            cls_to_use = random.sample(class_to_select, 2)

            inputs_u_s0, targets_u0 = strongTransform_class_mix(image_mix[0], images_target[0], label_mix[0],
                                                                targets_u_w[0], MixMask0_lam, MixMask0, gta5_cls_mixer, cls_to_use,
                                                                strong_parameters)
            inputs_t_s0, targets_t0 = strongTransform_class_mix(image_mix[0], images[0], label_mix[0], labels[0],
                                                                MixMask0_lam, MixMask0, gta5_cls_mixer, cls_to_use,
                                                                strong_parameters)

            inputs_u_s1, targets_u1 = strongTransform_class_mix(image_mix[1], images_target[1], label_mix[1],
                                                                targets_u_w[1], MixMask1_lam, MixMask1, gta5_cls_mixer, cls_to_use,
                                                                strong_parameters)
            inputs_t_s1, targets_t1 = strongTransform_class_mix(image_mix[1], images[1], label_mix[1], labels[1],
                                                                MixMask1_lam, MixMask1, gta5_cls_mixer, cls_to_use,
                                                                strong_parameters)


            inputs_u_s = torch.cat((inputs_u_s0, inputs_u_s1))
            inputs_t_s = torch.cat((inputs_t_s0, inputs_t_s1))


            p1,p2 = model(inputs_u_s)
            ap = nn.AdaptiveAvgPool2d((1,1))
            logits_u_s = interp(p1)
            gs = ap(p2).flatten(1)
            p2 = interp(p2)

            f_source = MixMask0_lam * p2
            f_source = ap(f_source).flatten(1)

            pt1,pt2 = model(inputs_t_s)
            logits_t_s = interp(pt1)
            gt = ap(pt2).flatten(1)
            pt2 = interp(pt2)
            f_target = MixMask0_lam * pt2
            f_target = ap(f_target).flatten(1)
            loss_feature = mmd_rbf(f_source,f_target)

            loss_global = mmd_rbf(gs,gt)
            loss_mmd = 0.005*loss_global + 0.005*loss_feature



            targets_u = torch.cat((targets_u0, targets_u1)).long()
            targets_t = torch.cat((targets_t0, targets_t1)).long()


            L_l2 = loss_calc(logits_t_s, labels) * (1-lam) + lam * loss_calc(logits_t_s, targets_t)


            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
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
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':

                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight_a)
                + consistency_weight * unlabeled_loss(logits_u_s, targets_u_w, pixelWiseWeight_b)

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

        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)
        if i_iter % 100 == 0 :

            print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}, loss_2 = {4:.3f}, lambda = {5:.3f}, loss_mmd = {6:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value, loss_2_value,lam,loss_mmd_value))






        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            mIoU = evaluate(model, 'cityscapes', ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)

            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(mIoU,i_iter, model, optimizer, config, ema_model, save_best=True)

            print('The best miou is %.4f' % best_mIoU)


    #_save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    mIoU = evaluate(model, 'cityscapes', ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(mIoU,i_iter, model, optimizer, config, ema_model, save_best=True)



    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']

    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

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

    gpus = (0,1,2,3)[:args.gpus]

    main()

