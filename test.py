import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':

    img = Image.open('./test/night.jpg')
    T = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()])
    input = T(img).unsqueeze(0).cuda()
    model = torch.load('model.pkl')

    outputs = model(input)
    _, preds = torch.max(outputs, 1)
    plt.imshow(preds.squeeze().cpu())
    plt.show()
