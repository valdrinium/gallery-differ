import os, sys
import albumentations, imagehash

from cv2 import cv2
from PIL import Image
from munkres import Munkres
from multiprocessing import Pool, cpu_count

IMAGE_RESIZE_TARGET = 512  # so that we don't run out of memory
MAX_HAMMING_DIST = 64.0  # maximum possible hamming distance of any 2 64-bit hashes
PHASH_THRESHOLD = 12.0  # any hamming distance above this value is produced by different images in pHash's context
CROP_THRESHOLD = 0.09  # any hamming distance above this values produced by different images in cropResistantHash's context
MAX_ANGLE = 30  # maximum rotation angle supported; any image rotated above this value will not get matched with its regular version


def loadGallery(folder):
    resizePipeline = albumentations.Compose(
        [
            albumentations.Resize(
                IMAGE_RESIZE_TARGET,
                IMAGE_RESIZE_TARGET,
                interpolation=cv2.INTER_LANCZOS4,
            ),
        ],
    )

    gallery = []
    for filename in os.listdir(folder):
        content = cv2.imread(os.path.join(folder, filename))
        if content is not None:
            gallery.append(
                {
                    "filename": filename,
                    "content": resizePipeline(image=content)["image"],
                }
            )

    return gallery


def transformedGallery(gallery, transformPipeline=albumentations.Compose([])):
    return [
        {
            "filename": image["filename"],
            "content": Image.fromarray(transformPipeline(image=image["content"])["image"]),
        }
        for image in gallery
    ]


def hashDifferenceBetween(referenceImage, targetImage, hashFunction):
    return (float)(hashFunction(referenceImage) - hashFunction(targetImage))


def hammingMatrixOf(referenceGallery, targetGallery, hashFunction, maxAngle=MAX_ANGLE):
    targetPipelines = [
        *[
            albumentations.Compose(
                [
                    albumentations.Rotate(limit=(angle, angle), p=1.0),
                ]
            )
            for angle in range(-maxAngle, maxAngle + 5, 5)
        ],
        *[
            albumentations.Compose(
                [
                    albumentations.Rotate(limit=(angle, angle), p=1.0),
                    albumentations.HorizontalFlip(p=1.0),
                ]
            )
            for angle in range(-maxAngle, maxAngle + 5, 5)
        ],
    ]

    hashPool = Pool(cpu_count())
    hashMatrix = [
        [
            [
                hashPool.apply_async(
                    hashDifferenceBetween,
                    (referenceImage["content"], targetImage["content"], hashFunction),
                )
                for targetImage in transformedGallery(targetGallery, targetPipeline)
            ]
            for targetPipeline in targetPipelines
        ]
        for referenceImage in transformedGallery(referenceGallery)
    ]

    hashPool.close()
    hashPool.join()

    hammingMatrix = [
        [{"distance": MAX_HAMMING_DIST} for _ in range(len(targetGallery))]
        for _ in range(len(referenceGallery))
    ]
    for referenceIndex, referenceImage in enumerate(referenceGallery):
        for pipelineIndex, _ in enumerate(targetPipelines):
            for targetIndex, targetImage in enumerate(targetGallery):
                potentialDistance = hashMatrix[referenceIndex][pipelineIndex][targetIndex].get()
                currentDistance = hammingMatrix[referenceIndex][targetIndex]["distance"]
                if potentialDistance < currentDistance:
                    hammingMatrix[referenceIndex][targetIndex] = {
                        "distance": potentialDistance,
                        "reference": referenceImage["filename"],
                        "target": targetImage["filename"],
                    }

    return hammingMatrix


def pHashMatch(referenceGallery, targetGallery):
    hammingMatrix = hammingMatrixOf(referenceGallery, targetGallery, imagehash.phash)
    munkresMatrix = [[column["distance"] for column in row] for row in hammingMatrix]
    optimalMatches = [
        hammingMatrix[row][column] for row, column in Munkres().compute(munkresMatrix)
    ]

    return list(filter(lambda match: match["distance"] <= PHASH_THRESHOLD, optimalMatches))


def colorHashWith8Binbits(image):
    return imagehash.crop_resistant_hash(
        image,
        hash_func=lambda image: imagehash.colorhash(image, binbits=8),
    )


def colorHashWith12Binbits(image):
    return imagehash.crop_resistant_hash(
        image,
        hash_func=lambda image: imagehash.colorhash(image, binbits=12),
    )


def averagedMatrices(matrices):
    if len(matrices) == 0:
        return []

    result = matrices[0]
    for matrix in matrices[1:]:
        for rowIndex, row in enumerate(matrix):
            for columnIndex, column in enumerate(row):
                targetColumn = result[rowIndex][columnIndex]
                targetColumn["distance"] = targetColumn["distance"] + column["distance"]
    for row in result:
        for column in row:
            column["distance"] = column["distance"] / len(matrices)

    return result


def cropResistantMatch(referenceGallery, targetGallery):
    if len(referenceGallery) == 0 or len(targetGallery) == 0:
        return []

    hashingStrategies = [colorHashWith8Binbits, colorHashWith12Binbits]
    hammingMatrices = [
        hammingMatrixOf(
            referenceGallery,
            targetGallery,
            hashingStrategy,
            0,
        )
        for hashingStrategy in hashingStrategies
    ]

    averagedHammingMatrix = averagedMatrices(hammingMatrices)
    munkresMatrix = [[column["distance"] for column in row] for row in averagedHammingMatrix]
    optimalMatches = [
        averagedHammingMatrix[row][column] for row, column in Munkres().compute(munkresMatrix)
    ]

    return list(filter(lambda match: match["distance"] <= CROP_THRESHOLD, optimalMatches))


def galleriesWithoutMatches(referenceGallery, targetGallery, matches):
    for match in matches:
        referenceGallery = list(
            filter(lambda image: image["filename"] != match["reference"], referenceGallery)
        )
        targetGallery = list(
            filter(lambda image: image["filename"] != match["target"], targetGallery)
        )

    return (referenceGallery, targetGallery)


def generateChangelist(referenceGallery, targetGallery, matches, solvedBy, isFinal=True):
    changelist = []
    for match in matches:
        change = match.copy()
        change["resolution"] = "unchanged" if change["distance"] == 0.0 else "light changes"
        change["solvedBy"] = solvedBy

        changelist.append(change)

    if isFinal:
        for referenceImage in referenceGallery:
            changelist.append(
                {
                    "reference": referenceImage["filename"],
                    "resolution": "removed",
                }
            )
        for targetImage in targetGallery:
            changelist.append(
                {
                    "target": targetImage["filename"],
                    "resolution": "added",
                }
            )

    return changelist


if __name__ == "__main__":
    if (
        not len(sys.argv) == 3
        or not os.path.isdir(sys.argv[1])
        or not os.path.isdir(sys.argv[2])
    ):
        print(
            'Usage: time python src/matching/index.py "/code/samples/002 - gin/original" "/code/samples/002 - gin/attack001"'
        )
        exit(-1)

    referenceFolder = os.path.normpath(sys.argv[1])
    targetFolder = os.path.normpath(sys.argv[2])

    referenceGallery = loadGallery(referenceFolder)
    targetGallery = loadGallery(targetFolder)

    pHashMatches = pHashMatch(referenceGallery, targetGallery)
    referenceGallery, targetGallery = galleriesWithoutMatches(
        referenceGallery, targetGallery, pHashMatches
    )
    changelist = generateChangelist(
        referenceGallery, targetGallery, pHashMatches, "pHash", False
    )

    cropResistantHashes = cropResistantMatch(referenceGallery, targetGallery)
    referenceGallery, targetGallery = galleriesWithoutMatches(
        referenceGallery, targetGallery, cropResistantHashes
    )
    changelist = changelist + generateChangelist(
        referenceGallery, targetGallery, cropResistantHashes, "cropResistantHash"
    )

    print(changelist)

    # TODO: should also test how it handles watermarks
    # TODO: will not work when crop is combined with others
