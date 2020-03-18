import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class Evaluation(object):

    def __init__(self, n_class):
        self.class_name = ['unlabeled', 'ego vehicle', 'rectified border',
                           'out of roi', 'static', 'dynamic', 'ground', 'road',
                           'sidewalk', 'parking', 'rail track', 'building', 'wall',
                           'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
                           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                           'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
                           'train', 'motorcycle', 'bicycle']
        np.seterr(divide='ignore', invalid='ignore')
        self.n_class = n_class
        self.hist = np.zeros((n_class, n_class))

    def record_result(self, label_trues, label_preds):
        """Record the result of each batch"""
        for lt, lp in zip(label_trues, label_preds):
            self.hist += _fast_hist(lt.flatten(), lp.flatten(), self.n_class)

    def get_eval_result(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        """
        # true positive / (true positive + false positive)
        self.mean_acc = np.diag(self.hist).sum() / self.hist.sum()

        # accuracy of each class
        self.acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        self.mean_acc_cls = np.nanmean(self.acc_cls)

        # IoU
        self.iou_cls = np.diag(self.hist) / (self.hist.sum(axis=1) +
                                             self.hist.sum(axis=0) - np.diag(self.hist))
        self.mean_iou = np.nanmean(self.iou_cls)

        # Frequency Weighted Intersection over Union
        freq = self.hist.sum(axis=0) / self.hist.sum()
        self.fwiou = (freq[freq > 0] * self.iou_cls[freq > 0]).sum()

    def clear_record(self):
        """
        clear the recorded result
        """
        self.hist = np.zeros((self.n_class, self.n_class))


if __name__ == '__main__':
    a1 = np.arange(301056).reshape(6, 224, 224)
    a2 = np.arange(301056).reshape(6, 224, 224)

    eval1 = Evaluation(10)
    eval1.record_result(a1, a2)
    eval1.get_eval_result()
    print(eval1.mean_iou)
    print(eval1.hist)
    eval1.clear_record()
    print(eval1.hist)
