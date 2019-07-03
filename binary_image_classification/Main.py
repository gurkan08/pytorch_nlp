import torch
import os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd

from Optimizer import Optimizer
from Loss import Loss
from QualityDataset import QualityDataset
from Network import Network
from Utility import Utility


class Quality(object):
    def __init__(self, args):

        self.INPUT_DIR = args.input_dir
        self.TRAIN_DIR = os.path.join(self.INPUT_DIR, "train")
        self.VALID_DIR = os.path.join(self.INPUT_DIR, "valid")
        self.MODEL_DIR = os.path.join(self.INPUT_DIR, "model")
        self.PLOT_DIR = os.path.join(self.INPUT_DIR, "plot")
        self.MISCLASS_DIR = os.path.join(self.INPUT_DIR, "misclass")

        self.BATCH_SIZE = int(args.batch)
        self.WIDTH = int(args.width)
        self.HEIGHT = int(args.height)
        self.DROPOUT = float(args.drop)
        self.LOSS = int(args.loss)
        self.OPTIMIZER = int(args.opt)
        self.LEARNING_RATE = float(args.lr)
        self.EPOCH = int(args.epoch)
        self.NETWORK = int(args.n)
        self.USE_CUDA = bool(args.cuda) and torch.cuda.is_available()
        self.SAVE_PERIOD = int(args.save)
        self.BINARY_CLASS_THRESHOLD = float(args.th)
        self.SAVE_MODEL = args.save_model

        EXP_FOLDER_NAME = str(self.NETWORK) + "_" + str(self.EPOCH) + "_" + str(self.BATCH_SIZE) + "_" + str(
            self.LEARNING_RATE)[2:] + "_" + str(self.DROPOUT)
        self.MODEL_DIR = os.path.join(self.MODEL_DIR, EXP_FOLDER_NAME)
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)

        self.PLOT_DIR = os.path.join(self.PLOT_DIR, EXP_FOLDER_NAME)
        if not os.path.exists(self.PLOT_DIR):
            os.makedirs(self.PLOT_DIR)

        self.MISCLASS_DIR = os.path.join(self.MISCLASS_DIR, EXP_FOLDER_NAME)
        if not os.path.exists(self.MISCLASS_DIR):
            os.makedirs(self.MISCLASS_DIR)

        # create network
        self.NET = self.get_network()
        net_total_params = sum(p.numel() for p in self.NET.parameters())
        print("#net params:", net_total_params)
        if self.USE_CUDA:
            self.NET.cuda()

    def get_network(self):
        if self.NETWORK == 1:
            return Network(self.USE_CUDA, self.DROPOUT)

    def save_model(self, file_name):
        torch.save(self.NET.state_dict(), os.path.join(self.MODEL_DIR, file_name + ".pth"))

    def save_plot(self, file_name, epoch, args):
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(np.array(range(1, epoch + 1)), np.array(args[0]), "g", label="train")
        ax[0].plot(np.array(range(1, epoch + 1)), np.array(args[1]), "r", label="valid")
        ax[0].set(xlabel="epoch", ylabel="loss")
        ax[0].legend(loc="upper right")

        ax[1].plot(np.array(range(1, epoch + 1)), np.array(args[2]), "g", label="train")
        ax[1].plot(np.array(range(1, epoch + 1)), np.array(args[3]), "r", label="valid")
        ax[1].set(xlabel="epoch", ylabel="acc")
        ax[1].legend(loc="lower right")

        fig.savefig(os.path.join(self.PLOT_DIR, file_name + ".png"))
        plt.close(fig)

    def save_misclassified(self, misclassified, filename, select="train"):
        save_dir = os.path.join(self.MISCLASS_DIR, select)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataframe = pd.DataFrame(misclassified, columns=["image", "prob.", "pred_label", "gt_label"])
        pd_writer = pd.ExcelWriter(os.path.join(save_dir, filename) + ".xlsx", engine='xlsxwriter')
        dataframe.to_excel(pd_writer)
        pd_writer.save()

    def __train(self, train_loader, valid_loader, optimizer, criterion, scheduler):
        plot_train_loss = []
        plot_valid_loss = []
        plot_train_acc = []
        plot_valid_acc = []
        for epoch in range(1, self.EPOCH + 1):
            print("*" * 20)
            print("epoch:{}".format(epoch))

            s_t = time.time()
            train_loss, cm, train_acc, fpr, fnr, train_misclassified = self.train(train_loader, optimizer, criterion)
            t = Utility.elapsed_time(s_t, time.time())

            plot_train_loss.append(train_loss)
            plot_train_acc.append(train_acc)
            Quality.information(train_loss, cm, train_acc, fpr, fnr, t, mode="train")

            s_t = time.time()
            valid_loss, cm, valid_acc, fpr, fnr, valid_misclassified = self.valid(valid_loader, criterion)
            t = Utility.elapsed_time(s_t, time.time())
            plot_valid_loss.append(valid_loss)
            plot_valid_acc.append(valid_acc)
            Quality.information(valid_loss, cm, valid_acc, fpr, fnr, t, mode="valid")

            if not epoch % self.SAVE_PERIOD:
                file_name = str(self.NETWORK) + "_" + str(epoch) + "_" + str(self.BATCH_SIZE) + "_" + str(
                    self.LEARNING_RATE)[2:] + "_" + str(self.DROPOUT)
                if self.SAVE_MODEL:
                    self.save_model(file_name)
                args = [plot_train_loss,
                        plot_valid_loss,
                        plot_train_acc,
                        plot_valid_acc
                        ]
                self.save_plot(file_name, epoch, args)
                self.save_misclassified(train_misclassified, file_name, select="train")
                self.save_misclassified(valid_misclassified, file_name, select="valid")

            scheduler.step()
            # self.adjust_lr(optimizer,self.LEARNING_RATE,epoch)

    def information(loss, cm, acc, fpr, fnr, t, mode="train"):
        print("mode:{}, time:{}".format(mode, t))
        print("loss: ", loss)
        print("acc: ", acc)
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("cm: ", cm)

    def get_confusion_matrix(gt_labels, pred_labels):
        cm = confusion_matrix(gt_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tn + fp + fn + tp)
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        return cm, acc, fpr, fnr

    def train(self, train_loader, optimizer, criterion):
        self.NET.train()
        train_loss = []
        all_pred_labels = []
        all_gt_labels = []
        misclassified = []
        for _, (img, label, img_name) in enumerate(train_loader):
            if self.USE_CUDA:
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            else:
                img = Variable(img)
                label = Variable(label)

            out = self.NET.net_forward(img)
            loss = criterion(out, label.float())
            train_loss.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prob = out.data.cpu().numpy().flatten()
            label = label.data.cpu().numpy().flatten()
            pred_labels = np.array([1 if x > self.BINARY_CLASS_THRESHOLD else 0 for x in prob])
            all_gt_labels.append(label)
            all_pred_labels.append(pred_labels)

            # misclassified samples
            for idx, x in enumerate(label):
                if label[idx] != pred_labels[idx]:
                    misclassified.append((img_name[idx], prob[idx], pred_labels[idx], label[idx]))

        cm, acc, fpr, fnr = Quality.get_confusion_matrix(np.concatenate(all_gt_labels).ravel(),
                                                         np.concatenate(all_pred_labels).ravel()
                                                         )
        return np.mean(train_loss), cm, acc, fpr, fnr, misclassified

    def valid(self, valid_loader, criterion):
        self.NET.eval()
        valid_loss = []
        all_pred_labels = []
        all_gt_labels = []
        misclassified = []
        for _, (img, label, img_name) in enumerate(valid_loader):
            if self.USE_CUDA:
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            else:
                img = Variable(img)
                label = Variable(label)

            out = self.NET.net_forward(img)
            loss = criterion(out, label.float())
            valid_loss.append(loss.data[0])

            prob = out.data.cpu().numpy().flatten()
            label = label.data.cpu().numpy().flatten()
            pred_labels = np.array([1 if x > self.BINARY_CLASS_THRESHOLD else 0 for x in prob])
            all_gt_labels.append(label)
            all_pred_labels.append(pred_labels)

            # misclassified samples
            for idx, x in enumerate(label):
                if label[idx] != pred_labels[idx]:
                    misclassified.append((img_name[idx], prob[idx], pred_labels[idx], label[idx]))

        cm, acc, fpr, fnr = Quality.get_confusion_matrix(np.concatenate(all_gt_labels).ravel(),
                                                         np.concatenate(all_pred_labels).ravel()
                                                         )
        return np.mean(valid_loss), cm, acc, fpr, fnr, misclassified

    def read_dataset(self):
        train_loader = DataLoader(dataset=QualityDataset(self.TRAIN_DIR, self.WIDTH, self.HEIGHT),
                                  batch_size=self.BATCH_SIZE,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=QualityDataset(self.VALID_DIR, self.WIDTH, self.HEIGHT),
                                  batch_size=self.BATCH_SIZE,
                                  shuffle=True)
        return train_loader, valid_loader

    def start(self):
        train_loader, valid_loader = self.read_dataset()
        optimizer = Optimizer.get_optimizer(self.OPTIMIZER, self.NET, self.LEARNING_RATE)
        criterion = Loss.get_loss(self.LOSS)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        self.__train(train_loader, valid_loader, optimizer, criterion, scheduler)

    """
    def adjust_lr(self,optimizer,lr,epoch):
        for params in optimizer.param_groups:
            params["lr"]=float(params["lr"])*0.1
            self.LEARNING_RATE=float(params["lr"])
        print("new_lr->",self.LEARNING_RATE)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality DL code, (Gurkan Sahin, 07/05/2019)")
    parser.add_argument('-input_dir', help='Input directory', required=True)
    parser.add_argument('-epoch', help='Epoch size', required=True)
    parser.add_argument('-batch', help='Batch size', required=True)
    parser.add_argument('-lr', help='Learning rate (1e-4,1e-5)', required=True)
    parser.add_argument('-drop', help='Dropout rate', required=True)

    parser.add_argument('-opt', help='Optimizer (1:adam, 2:SGD default=1)', default=1)
    parser.add_argument('-loss', help='Loss function (1:BCELoss, default=1)', default=1)
    parser.add_argument('-n', help='Network (default=1)', default=1)
    parser.add_argument('-th', help='binary class.threshold', default=0.5)
    parser.add_argument('-save', help='save model frequency (default:1 epoch)', default=1)
    parser.add_argument('-save_model', help='save model flag', default=False)
    parser.add_argument('-cuda', help='use cuda (True/False)', default=True)
    parser.add_argument('-width', help='Image width size (default= 296)', default=296)  # 296
    parser.add_argument('-height', help='Image height size (default= 134)', default=134)  # 134

    args = parser.parse_args()
    print("all args:", args)

    obj = Quality(args)
    obj.start()







