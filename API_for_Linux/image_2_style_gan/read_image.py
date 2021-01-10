from PIL import Image
from torchvision import transforms


def image_reader_color(img_path,resize=None):
    with open(img_path,"rb") as f:
        image=Image.open(f)
        image=image.convert("RGB")
    if resize!=None:
        image=image.resize((resize,resize))
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    image = transform(image)

    image=image.unsqueeze(0)

    return image


def image_reader_gray(img_path,resize=None):
    with open(img_path,"rb") as f:
        image=Image.open(f)
        image=image.convert("L")
    if resize!=None:
        image=image.resize((resize,resize))
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    image = transform(image)

    image=image.unsqueeze(0)

    return image