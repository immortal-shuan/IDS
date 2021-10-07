import os
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

random.seed(102)


def train(model, optimizer, criteria, train_iter, dev_iter, output_dir, device, epoch_num):
    model.zero_grad()

    dev_acc = 0.0
    max_acc_index = 0
    dev_pre = 0.0
    dev_rec = 0.0
    dev_f1 = 0.0
    for epoch in range(epoch_num):

        train_preds = []
        train_labels = []

        train_step = 0.0
        train_loss = 0.0

        for feature_input, label in train_iter:
            model.train()
            feature_input, label = feature_input.to(device), label.to(device)

            pre_pro = model(feature_input)
            loss = criteria(pre_pro, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            train_loss += loss.item()
            pred = (pre_pro.cpu().argmax(-1) + 0).tolist()
            train_preds.extend(pred)
            train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_acc = accuracy_score(np.array(train_labels), np.array(train_preds))
        train_confusion = confusion_matrix(np.array(train_labels), np.array(train_preds))
        train_pre = precision_score(np.array(train_labels), np.array(train_preds), average='macro')
        train_rec = recall_score(np.array(train_labels), np.array(train_preds), average='macro')
        train_f1 = f1_score(np.array(train_labels), np.array(train_preds), average='macro')

        print('epoch:{}'.format(epoch))
        print('train_confusion:\n{}'.format(train_confusion))
        print('train_loss:{}  train_acc:{}  train_pre:{}  train_rec:{}  train_f1:{}'.format(
            train_loss/train_step, train_acc, train_pre, train_rec, train_f1)
        )

        dev_acc_, dev_pre_, dev_rec_, dev_f1_ = dev(model, dev_iter, device)
        if dev_acc < dev_acc_:
            dev_acc = dev_acc_
            dev_pre = dev_pre_
            dev_rec = dev_rec_
            dev_f1 = dev_f1_
            max_acc_index = epoch

            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    file = open('result7.txt', 'a')
    file.write('max_acc: {}, {}'.format(max_acc_index, dev_acc) + '\n')
    file.write('max_pre: {}'.format(dev_pre) + '\n')
    file.write('max_rec: {}'.format(dev_rec) + '\n')
    file.write('max_f1: {}'.format(dev_f1) + '\n' + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_acc: {}, {}'.format(max_acc_index, dev_acc))
    print('max_pre: {}'.format(dev_pre))
    print('max_rec: {}'.format(dev_rec))
    print('max_f1: {}'.format(dev_f1))
    print('-----------------------------------------------------------------------------------------------------------')


def dev(model, data_iter, device):
    model.eval()

    dev_preds = []
    dev_labels = []

    with torch.no_grad():
        for feature_input, label in data_iter:

            feature_input, label = feature_input.to(device), label.to(device)
            pre_pro = model(feature_input)
            pred = (pre_pro.cpu().argmax(-1) + 0).tolist()
            dev_preds.extend(pred)
            dev_labels.extend(label.cpu().tolist())

    dev_acc = accuracy_score(np.array(dev_labels), np.array(dev_preds))
    dev_confusion = confusion_matrix(np.array(dev_labels), np.array(dev_preds))
    dev_pre = precision_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_rec = recall_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_f1 = f1_score(np.array(dev_labels), np.array(dev_preds), average='macro')

    print('dev_confusion:\n{}'.format(dev_confusion))
    print(
        'dev_acc:{}  dev_pre:{}  dev_rec:{}  dev_f1:{}'.format(
        dev_acc, dev_pre, dev_rec, dev_f1
        )
    )
    return dev_acc, dev_pre, dev_rec, dev_f1





