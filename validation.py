import torch
import data_loader
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import evaluation

BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_VAL = 3
USE_GPU = True


if __name__ == '__main__':

    transformed_data = data_loader.CityScape(train=False, rand=1)

    dataloaders = DataLoader(transformed_data, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    model = torch.load('model.pkl')
    evaluator_val = evaluation.Evaluation(35)
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

            result = preds.cpu()
            gt = labels

            for g, r in zip(gt, result):

                evaluator_val.record_result(g.numpy(), r.numpy())
            #     grid = torch.Tensor(2, 1, 224, 448)
            #     grid[0] = result[i]
            #     grid[1] = gt[i]
            #     grid = utils.make_grid(grid).numpy().transpose((1, 2, 0))

            #     plt.figure(0)
            #     plt.imshow(grid[:, :, 0])
            #     plt.colorbar()

            #     # plt.imshow(preds[0].cpu())
            #     plt.axis('off')
            #     plt.show()
            #     images_so_far += 1

            #     if images_so_far >= NUM_VAL:
            #         break

            # if images_so_far >= NUM_VAL:
            #     break

        evaluator_val.get_eval_result()
        print('\n*****overall result*****')
        print('mean_accuracy:{:.4f}, mean_iou:{:.4f}, frquency_weighted_iou:{:.4f}'.format(
            evaluator_val.mean_acc, evaluator_val.mean_iou, evaluator_val.fwiou))
        print('\n*****class result*****')
        for c, m1, m2 in zip(evaluator_val.class_name, evaluator_val.acc_cls, evaluator_val.iou_cls):
            print('class:{:18s} | accuracy:{:.3f} | iou:{:.3f}'.format(
                c, m1, m2))
