import model
import data_loader
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import time
import sys
import evaluation
import copy

LEARNING_RATE = 0.00005
EPOCH_NUM = 1000
BATCH_SIZE = 6
NUM_WORKERS = 2
USE_GPU = True
USE_PRE_TRAIN = True
CHECKPONT = 10

if __name__ == '__main__':

    if USE_PRE_TRAIN:
        net = torch.load('model.pkl')
    else:
        net = model.FCN(34)  # 34 classes for Cityscape Dataset

    if USE_GPU:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    evaluator = evaluation.Evaluation(34)

    transformed_data = data_loader.CityScape(rand=0.5)
    dataloaders = DataLoader(transformed_data, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS)
    evaluation_data = data_loader.CityScape(train=False, rand=-1)
    dataloaders_eval = DataLoader(evaluation_data, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)

    loss_plt = []
    epoch_plt = []
    loss_eval_plt = []
    plt.ion()
    start_time = time.time()
    for i in range(EPOCH_NUM):
        # forward pass
        running_loss = 0.0
        running_loss_eval = 0.0
        evaluator.clear_record()
        epoch_start_time = time.time()
        # iterate the data
        for batches in dataloaders:
            batch_start_time = time.time()
            if USE_GPU:
                inputs = batches['image'].cuda()
#                labels = batches['label'].cuda()
            else:
                inputs = batches['image']
#                labels = batches['label']
            load_data_time = time.time()

            # zero the gradient
            forward_start_time = time.time()
            optimizer.zero_grad()
            outputs = net(inputs)
            forward_end_time = time.time()

            if USE_GPU:
                labels = batches['label'].cuda()
            else:
                labels = batches['label']

            backward_start_time = time.time()
            loss = criterion(outputs, labels)
            loss.backward()
            backward_end_time = time.time()

            optimizer.step()
            # lr_scheduler.step()

            # get the loss of each epoch
            running_loss += loss.item() * labels.size(0)

        # evaluation
        with torch.no_grad():
            for batches in dataloaders_eval:

                if USE_GPU:
                    inputs_eval = batches['image'].cuda()
                    labels_eval = batches['label'].cuda()
                else:
                    inputs_eval = batches['image']
                    labels_eval = batches['label']

                outputs_eval = net(inputs_eval)
                loss_eval = criterion(outputs_eval, labels_eval)
                running_loss_eval += loss_eval.item() * labels_eval.size(0)

                _, preds_eval = torch.max(outputs_eval, 1)
                for g, r in zip(labels_eval.cpu(), preds_eval.cpu()):
                    evaluator.record_result(g.numpy(), r.numpy())
        evaluator.get_eval_result()

        # visualize the process
        loss_plt.append(running_loss/len(transformed_data))
        epoch_plt.append(i + 1)
        loss_eval_plt.append(running_loss_eval/len(evaluation_data))
        if i+1 > 1:
            plt.cla()
            plt.plot(epoch_plt, loss_plt, 'r-', lw=5, label='train')
            plt.plot(epoch_plt, loss_eval_plt, 'b-', lw=5, label='val')
            plt.legend(loc=0, ncol=2)
            plt.title('Loss=%.4f' % loss_plt[-1],
                      fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
        print('***** epoch {:d} *****'.format(i+1))
        print('Loss: {:.4f}'.format(loss_plt[-1]))
        print('runtime of this epoch: {:.4f} s'.format(
            time.time()-epoch_start_time))
        print('load data time: {:.4f} s'.format(
            load_data_time-batch_start_time))
        print('forward pass time: {:.4f} s'.format(
            forward_end_time-forward_start_time))
        print('backward pass time: {:.4f} s'.format(
            backward_end_time-backward_start_time))
        print('\n*****overall evaluation result*****')
        print('mean_accuracy:{:.4f}, mean_iou:{:.4f}, frquency_weighted_iou:{:.4f}'.format(
            evaluator.mean_acc, evaluator.mean_iou, evaluator.fwiou))
        print('*****end of epoch {:d}*****\n'.format(i+1))

        if (i+1) % CHECKPONT == 0:
            print("\a")
            print('\nDo you want to keep training???\n')
            decision = input('q:quit, else:yes\n')
            if decision == 'q':
                break

    end_time = time.time()

    plt.ioff()
    plt.show()
    torch.save(net, 'model.pkl')
    print('Training time: {:.4f}'.format(end_time-start_time))
