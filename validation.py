import torch
import data_loader
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_VAL = 3
USE_GPU = True


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':

    raw_data = data_loader.CityScape(train=False, all=False)
    transformed_data = []
    composed = transforms.Compose([data_loader.Rescale(256),
                                   data_loader.RandomCrop(224),
                                   data_loader.ToTensor()])
    for d in raw_data:
        transformed_data.append(composed(d))

    dataloaders = DataLoader(transformed_data, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS)

    model = torch.load('model.pkl')
    images_so_far = 0
    with torch.no_grad():

        for batches in dataloaders:

            if USE_GPU:
                inputs = batches['image'].cuda()
                labels = batches['label']
            else:
                inputs = batches['image']
                labels = batches['label']

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            result = preds.unsqueeze(1).cpu()
            gt = labels
                             
            for i in range(BATCH_SIZE):

                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(gt[i].numpy(), result[i].numpy(), 35)
                print('accuracy:{:.4f}, mean_iu:{:.4f}'.format(acc,mean_iu))
                
                grid = torch.Tensor(2, 1, 224, 224)
                grid[0] = result[i]
                grid[1] = gt[i]
                grid = utils.make_grid(grid).numpy().transpose((1, 2, 0))

                plt.figure(0)
                plt.imshow(grid[:, :, 0])
                plt.colorbar()

                # plt.imshow(preds[0].cpu())
                plt.axis('off')
                plt.show()
                images_so_far += 1

                if images_so_far >= NUM_VAL:
                    break

            if images_so_far >= NUM_VAL:
                break
