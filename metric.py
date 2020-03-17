import numpy as np

class_name = ['unlabeled', 'ego vehicle', 'rectified border',
              'out of roi', 'static', 'dynamic', 'ground', 'road',
              'sidewalk', 'parking', 'rail track', 'building', 'wall',
              'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
              'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
              'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train',
              'motorcycle', 'bicycle', 'license plate']


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def evaluation_result(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    np.seterr(divide='ignore', invalid='ignore')
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # true positive / (true positive + false positive)
    mean_acc = np.diag(hist).sum() / hist.sum()

    # accuracy of each class
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    mean_acc_cls = np.nanmean(acc_cls)

    # IoU
    iou_cls = np.diag(hist) / (hist.sum(axis=1) +
                               hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou_cls)

    # Frequency Weighted Intersection over Union
    freq = hist.sum(axis=0) / hist.sum()
    fwiou = (freq[freq > 0] * iou_cls[freq > 0]).sum()
    return mean_acc, acc_cls, mean_acc_cls, iou_cls, mean_iou, fwiou


if __name__ == '__main__':
    a1 = np.arange(301056).reshape(6, 224, 224)
    a2 = np.arange(301056).reshape(6, 224, 224)
    mean_acc, acc_cls, mean_acc_cls, iou_cls, mean_iou, fwiou = evaluation_result(
        a1, a2, 10)
    print(mean_acc)
    print(acc_cls)
    print(mean_acc_cls)
    print(iou_cls)
