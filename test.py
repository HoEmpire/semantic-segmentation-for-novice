import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':

    img = Image.open('./test/reference4.jpg')

    T = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor()])
    input = T(img).unsqueeze(0).cuda()
    model = torch.load('./bak/model.pkl')

    outputs = model(input)
    _, preds = torch.max(outputs, 1)
    plt.subplot(1, 2, 1)
    plt.imshow(input.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(preds.squeeze().cpu())
    plt.xticks([])
    plt.yticks([])
    plt.show()
