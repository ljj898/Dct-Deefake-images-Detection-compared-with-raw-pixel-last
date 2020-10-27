import time
import datetime
import traceback
import os
import cv2

import torch
from utils.Net import save_ckpt
import torch.nn.functional as F
import torch.nn as nn
# from models.ResNet import ResNet
from models.Vgg import Vgg16
from utils.metric import evaluate, confusion_matrix

from tqdm import tqdm


# from network._model import Discriminator


class Solver(object):
    def __init__(self, config, dataloader=None, val_dataloader=None):
        super(Solver, self).__init__()
        self.use_gpu = True if torch.cuda.is_available() else False
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.start_step = 0
        self.build_model()

        # Build tensorboard if use
        if self.config.use_tensorboard and self.config.mode == 'train':
            self.build_tensorboard()

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # Set the tensorboard logger
        self.tb_logger = SummaryWriter(self.config.log_dir)

    def build_model(self):
        ###if self.config.model == 'resnet50':
        # self.model = ResNet('resnet50', True, self.config.use_srm)
        if self.config.model == 'vgg':
            self.model = Vgg16(False, self.config.use_srm)
        # if self.config.model == 'D':
        # self.model = Discriminator(256)

        if self.config.mode == 'train':
            # Optimizer
            if self.config.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
            elif self.config.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=self.config.lr, momentum=self.config.momentum,
                                                 weight_decay=self.config.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.config.lr_decay_step,
                                                                gamma=self.config.lr_decay_gamma)
            #self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
            # Total loss
            self.cel = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self):
        # setting to train mode
        self.model.train()

        if self.config.resume_iter:
            checkpoint_path = os.path.join(self.config.save_dir, 'ckpt',
                                           'model_' + str(self.config.resume_iter) + '.pth')
            print('Loading checkpoint %s' % checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        start_time = time.time()

        if self.config.resume_iter:
            self.start_step = self.config.resume_iter

        data_iter = iter(self.dataloader)
        try:
            for step in range(self.start_step, self.config.max_iter):

                try:
                    imgs, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    imgs, labels = next(data_iter)

                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                pred = self.model(imgs)

                total_loss = self.cel(pred, labels.long())
                # backward
                self.optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-4)
                self.optimizer.step()
                self.lr_scheduler.step()
                # display results
                if (step + 1) % self.config.disp_interval == 0:
                    loss = {}
                    loss['S/total_loss'] = total_loss.item()

                    lr = {}
                    lr['lr'] = self.optimizer.param_groups[0]["lr"]
                    acc = {}
                    # validate during trainning
                    acc['acc'] = self.validate()
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    log = 'time cost: {} iter: {} / {}' \
                        .format(elapsed, step + 1, self.config.max_iter)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    # Tensorboard logger
                    if self.config.use_tensorboard:
                        for tag, value in loss.items():
                            self.tb_logger.add_scalar(tag, value, step + 1)
                        for tag, value in lr.items():
                            self.tb_logger.add_scalar(tag, value, step + 1)
                        for tag, value in acc.items():
                            self.tb_logger.add_scalar(tag, value, step + 1)

                # save checkpoint
                if (step + 1) % self.config.save_interval == 0:
                    state = {
                        'step': step + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'model': self.model.state_dict()
                    }

                    save_ckpt(self.config.save_dir, state, step + 1)

                    # acc = {}
                    # acc['acc'] = self.validate()
                    # if self.config.use_tensorboard:



        except(RuntimeError, KeyboardInterrupt):
            del data_iter
            stack_trace = traceback.format_exc()
            print(stack_trace)
        finally:
            if self.config.use_tensorboard:
                self.tb_logger.close()

    def validate(self):
        self.model.eval()

        correct = 0
        total = 0
        for imgs, labels in self.val_dataloader:

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda().long()

            outputs = self.model(imgs)

            # predicted = outputs > 0.1
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # break
        accuracy = correct / total
        print('Accuracy of {} images: {}'.format(total, accuracy))
        self.model.train()

        return accuracy

    def test(self):
        self.model.eval()

        checkpoint_path = os.path.join(self.config.save_dir, 'ckpt', self.config.checkpoint)
        print('Loading checkpoint %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        del checkpoint
        torch.cuda.empty_cache()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_pos_scores = []
        for imgs, labels in self.val_dataloader:

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda().long()

            outputs = self.model(imgs)

            # predicted = outputs > 0.1

            _, predicted = torch.max(outputs, 1)
            outputs = F.softmax(outputs, dim=1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            outputs = outputs[:, -1].view(-1)
            all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            # break
        accuracy = correct / total
        acc, f1, roc_auc = evaluate(all_labels, all_preds, all_pos_scores)
        TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
        result = 'Total images:{},ACC:{},F1:{},ROC_AUC:{},TN:{},FN:{},TP:{},FP:{}' \
            .format( len(all_labels), acc, f1, roc_auc, TN, FN, TP, FP)
        print(result)
        print('Accuracy of {} images: {}'.format(total, accuracy))

    def val(self):
        self.model.eval()

        checkpoint_path = os.path.join(self.config.save_dir, 'ckpt', self.config.checkpoint)
        print('Loading checkpoint %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        del checkpoint
        torch.cuda.empty_cache()

        correct = 0
        total = 0
        for imgs, labels in self.val_dataloader:

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda().long()

            outputs = self.model(imgs)

            # predicted = outputs > 0.1
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # break
        accuracy = correct / total
        print('Accuracy of {} images: {}'.format(total, accuracy))
