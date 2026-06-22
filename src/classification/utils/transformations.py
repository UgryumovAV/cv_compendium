from torchvision import transforms


def _get_mnist_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    transformations_train = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    transformations_test = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    return transformations_train, transformations_test


def _get_imagenete_transforms():
    transformations_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.RandomPosterize(bits=2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transformations_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transformations_train, transformations_test


def get_transforms(
        model_name: str
) -> tuple[transforms.Compose, transforms.Compose]:
    if model_name in ['lenet']:
        return _get_mnist_transforms()
    elif model_name in ['alexnet']:
        return _get_imagenete_transforms()
    else:
        raise ValueError(f'Unknown model: {model_name}')

