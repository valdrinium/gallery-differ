# Testing puposes

import albumentations, os, sys

from cv2 import cv2


def loadImagesFromFolder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append((filename, image))

    return images


def transformImages(images):
    transformed = []
    for image in images:
        transformed.append((image[0], transformPipeline(image=image[1])["image"]))

    return transformed


def generateNextAttackFolderName(folder):
    return "attack" + str(len(os.listdir(folder))).zfill(3)


def saveTransformedImages(images, attackFolder):
    os.makedirs(attackFolder, exist_ok=True)
    for image in images:
        cv2.imwrite(os.path.join(attackFolder, image[0]), image[1])


if len(sys.argv) != 2:
    print('Usage: python src/generation/index.py "/code/samples/001 - gin/original"')
    exit(-1)

size = 512
probability = 0.2
transformPipeline = albumentations.Compose(
    [
        albumentations.Resize(size, size),
        albumentations.Blur(p=probability),  # 2
        albumentations.CLAHE(p=probability),
        albumentations.ChannelShuffle(p=probability),
        albumentations.CoarseDropout(p=probability),
        albumentations.Downscale(scale_min=0.25, scale_max=0.75, p=probability),
        albumentations.Equalize(p=probability),  # 12
        albumentations.GridDistortion(p=probability),
        albumentations.HorizontalFlip(p=probability),
        albumentations.HueSaturationValue(p=probability),
        albumentations.ISONoise(p=probability),
        albumentations.MotionBlur(p=probability),  # 22
        albumentations.MultiplicativeNoise(p=probability),
        albumentations.OpticalDistortion(p=probability),
        albumentations.Posterize(p=probability),
        albumentations.RandomBrightness(p=probability),
        albumentations.RandomContrast(p=probability),  # 32
        albumentations.RandomGamma(p=probability),
        albumentations.RandomShadow(p=probability),
        albumentations.RandomSizedCrop(
            [int(size * 0.95), int(size * 0.95)], size, size, p=probability
        ),
        albumentations.RGBShift(p=probability),
        albumentations.Rotate(limit=15, p=probability),  # 42
        albumentations.ToGray(p=probability),
        albumentations.ToSepia(p=probability),
    ]
)

originalPath, originalFolder = os.path.split(os.path.normpath(sys.argv[1]))
originalImages = loadImagesFromFolder(os.path.join(originalPath, originalFolder))
transformedImages = transformImages(originalImages)
attackFolder = generateNextAttackFolderName(originalPath)
saveTransformedImages(transformedImages, os.path.join(originalPath, attackFolder))
