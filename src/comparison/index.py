# Testing puposes

import albumentations, imagehash, os, sys

from cv2 import cv2
from PIL import Image


def loadImagesFromFolder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append((filename, image))

    return images


def transformImages(images, transformPipeline):
    transformed = []
    for image in images:
        transformed.append((image[0], transformPipeline(image=image[1])["image"]))

    return transformed


def cv2ToPIL(images):
    converted = []
    for image in images:
        converted.append((image[0], Image.fromarray(image[1])))

    return converted


if len(sys.argv) != 3:
    print(
        'Usage: python src/comparison/index.py "/code/samples/001 - gin/original" "/code/samples/002 - gin/original"'
    )
    exit(-1)

firstFolder = os.path.normpath(sys.argv[1])
secondFolder = os.path.normpath(sys.argv[2])

size = 512
hashFunctions = [
    # best against (<= 2.0): blur, channel shuffle, coarse dropout, downscale, horizontal flip, hue saturation value, iso noise, motion blur, multiplicative noise, optical distortion, posterize, random brightness, random contrast, random gamma, rgb shift, to gray
    # so-so against (<= 4.0): clahe, to sepia
    # worst against (> 4.0): equalize, grid distortion, random shadow, random sized crop, rotate, RANDOM COMBINATIONS, DIFFERENT IMAGES
    # CONCLUSION: not usable due to low performance on recognizing different images
    # (
    #     "aHash",
    #     imagehash.average_hash,
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 6.0): blur, channel shuffle, coarse dropout, downscale, horizontal flip, hue saturation value, iso noise, motion blur, multiplicative noise, optical distortion, posterize, random contrast, random gamma, rgb shift, to gray, DIFFERENT IMAGES
    # so-so against (<= 12.0): clahe, equalize, random brightness, to sepia, RANDOM COMBINATIONS
    # worst against (> 12.0): grid distortion, random shadow, random sized crop, rotate
    # CONCLUSION: very good on on differentiating between attack images and new ones
    # (
    #     "pHash",
    #     imagehash.phash,
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #         # CONCLUSION: to be used together with such rotations when it says > 12.0 because it solves the rotate problem
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.Rotate(limit=(-10, -10), p=1.0),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.Rotate(limit=(-10, -10), p=1.0),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.Rotate(limit=(10, 10), p=1.0),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.Rotate(limit=(10, 10), p=1.0),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 2.0): blur, horizontal flip, iso noise, motion blur, posterize, random gamma, rgb shift, to gray
    # so-so against (<= 4.0): clahe, channel shuffle, coarse dropout, equalize, hue saturation value, multiplicative noise, optical distortion, random contrast, RANDOM COMBINATIONS, DIFFERENT IMAGES
    # worst against (> 4.0): downscale, grid distortion, random brightness, random shadow, random sized crop, rotate, to sepia
    # CONCLUSION: not usable due to low performance on differentiating between attack images and new ones
    # (
    #     "dHash",
    #     imagehash.dhash,
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 6.0):
    # so-so against (<= 12.0):
    # worst against (> 12.0):
    # CONCLUSION:
    # (
    #     "wHash - haar",
    #     imagehash.whash,
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 6.0):
    # so-so against (<= 12.0):
    # worst against (> 12.0):
    # CONCLUSION:
    # (
    #     "wHash - db4",
    #     lambda img: imagehash.whash(img, mode="db4"),
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 6.0):
    # so-so against (<= 12.0):
    # worst against (> 12.0):
    # CONCLUSION:
    # (
    #     "colorHash",
    #     imagehash.colorhash,
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #                 albumentations.HorizontalFlip(p=1.0),
    #             ]
    #         ),
    #     ],
    # ),
    # best against (<= 0.06): ~
    # so-so against (<= 0.09): ~
    # worst against (> 0.09): ~
    # CONCLUSION: to be used together with phash when it says > 12.0 because it solves the random sized crop problem
    # (
    #     "cropResistantHash",
    #     lambda img: imagehash.crop_resistant_hash(img, hash_func=imagehash.colorhash),
    #     [
    #         albumentations.Compose(
    #             [
    #                 albumentations.Resize(size, size),
    #             ]
    #         ),
    #     ],
    # ),
]

firstImageList = loadImagesFromFolder(firstFolder)
secondImageList = loadImagesFromFolder(secondFolder)

for hashFunction in hashFunctions:
    name = hashFunction[0]
    hasher = hashFunction[1]
    transformPipelines = hashFunction[2]

    results = [64.0] * len(firstImageList)
    firstImageSet = cv2ToPIL(transformImages(firstImageList, transformPipelines[0]))
    for transformPipeline in transformPipelines:
        secondImageSet = cv2ToPIL(transformImages(secondImageList, transformPipeline))
        if len(firstImageSet) != len(secondImageSet):
            print("Variable-length galleries not yet supported")
            exit(-1)

        for index in range(len(firstImageSet)):
            firstImage = firstImageSet[index][1]
            secondImage = secondImageSet[index][1]

            total = 0
            retries = 3
            for _ in range(retries):
                firstHash = hasher(firstImage)
                secondHash = hasher(secondImage)

                total = total + (firstHash - secondHash)

            results[index] = min(results[index], total / retries)

    for index in range(len(firstImageList)):
        firstFilename = firstImageSet[index][0]
        secondFilename = secondImageSet[index][0]

        print(
            f"Using {name}, {firstFilename} and {secondFilename} differ by {results[index]}"
        )

    print()
