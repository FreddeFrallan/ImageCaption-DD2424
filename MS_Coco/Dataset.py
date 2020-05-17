import json, os


def getAllImagePaths(subFolder="train2014"):
    basePath = "MS_Coco/{}/".format(subFolder)
    return [(basePath + f, f) for f in os.listdir(basePath)]


def getAllInceptionFiles():
    basePath = "MS_Coco/InceptionFiles/"
    return [basePath + f for f in os.listdir(basePath)]


def getAll3DInceptionFiles():
    basePath = "MS_Coco/3D-Inception/"
    return [basePath + f for f in os.listdir(basePath)]


def getTraining3DInceptionFiles():
    basePath = "MS_Coco/3D-Inception/"
    return [basePath + f for f in os.listdir(basePath) if f != "MS-COCO-Train-3DInception-82768.pkl"]


def getValidation3DInceptionFiles():
    basePath = "MS_Coco/3D-Inception/"
    return basePath + "MS-COCO-Train-3DInception-82768.pkl"


def _loadAnnotations(captionsFile="captions_train2014.json"):
    basePath = "MS_Coco/"
    with open(basePath + "annotations/{}".format(captionsFile), 'r') as f:
        return json.load(f)


def getImgName2ImgId(captionsFile="captions_train2014.json"):
    annotations = _loadAnnotations(captionsFile)
    prefix = "COCO_train2014_" if captionsFile == "captions_train2014.json" else ""

    imgIds = {}
    for annot in annotations['annotations']:
        image_id = annot['image_id']
        imgName = prefix + '%012d.jpg' % (image_id)
        imgIds[imgName] = image_id
    return imgIds, {v: k for k, v in imgIds.items()}


def getAllAnnotations(captionsFile="captions_train2014.json"):
    annotations = _loadAnnotations(captionsFile)

    all_captions = []
    captionToImg = {}

    prefix = "COCO_train2014_" if captionsFile == "captions_train2014.json" else ""
    for annot in annotations['annotations']:
        # caption = '<start> ' + annot['caption'] + ' <end>'
        caption = annot['caption'] + ' <end>'
        # caption = annot['caption'] + ' <end>'
        image_id = annot['image_id']
        imgName = prefix + '%012d.jpg' % (image_id)

        captionToImg[caption] = imgName
        all_captions.append(caption)

    return all_captions, captionToImg
