from MS_Coco import Dataset
import CaptionModel, Utils
import TextPreProcessing
import tensorflow as tf
import numpy as np
import pickle, os


def _loadInceptionFile(filePath, shuffle=True):
    with open(filePath, 'rb') as fp:
        data = pickle.load(fp)
    keys = list(data.keys())
    if (shuffle):
        np.random.shuffle(keys)
    return data, keys


def _selectRandomCaption(captions):
    return captions[np.random.choice(range(len(captions)))]


def generateFetchData(fetchKeys, data, img2Captions):
    imgs, texts = [], []
    for k in fetchKeys:
        if (k in img2Captions):
            imgs.append(data[k])
            txt = _selectRandomCaption(img2Captions[k])
            texts.append(txt)

    f = lambda x: tf.convert_to_tensor(x)
    return f(imgs), f(texts)


def trainOnInceptionFile(filePath, img2Captions, model, batchSize=16, epoch='0', training=True):
    data, keys = _loadInceptionFile(filePath)
    windowSize = 100
    lossWindow = []
    allLosses = []
    for i in range(0, len(keys), batchSize):
        fetchKeys = keys[i:i + batchSize]
        inImgs, inTexts = generateFetchData(fetchKeys, data, img2Captions)
        _trainingStep(model, inImgs, inTexts, training, allLosses, lossWindow, windowSize, epoch)

    return allLosses


def trainOnImagesInInceptionFile(filePath, img2Captions, model, training=True, imgsMainPath="MS_Coco/train2014/",
                                 epoch=""):
    data, keys = _loadInceptionFile(filePath)
    imgData = [(f, imgsMainPath + f) for f in os.listdir(imgsMainPath) if f in keys]
    print("Training on {} images".format(len(imgData)))
    trainOnImages(imgData, model, img2Captions, training=training, epoch=epoch)


def trainOnImages(imgData, model, img2Captions, batchSize=16, training=True, epoch=""):
    f = lambda x: tf.convert_to_tensor(x)
    windowSize = 100
    lossWindow = []
    allLosses = []
    for i in range(0, len(imgData), batchSize):
        print(i, "/", len(imgData))
        batchImgs = imgData[i:i + batchSize]
        inImgs = Utils.loadImages([p for f, p in batchImgs if f in img2Captions])
        inTexts = f([_selectRandomCaption(img2Captions[f]) for f, p in batchImgs if f in img2Captions])
        _trainingStep(model, inImgs, inTexts, training, allLosses, lossWindow, windowSize, epoch)
    return allLosses


def _trainingStep(model, inImgs, inTexts, training, allLosses, lossWindow, windowSize, epoch):
    if (len(inImgs) > 0):
        loss, std = model.train_step(inImgs, inTexts, training=training)

        allLosses.append(loss)
        lossWindow.insert(0, loss.numpy())
        lossWindow = lossWindow[:windowSize]
        print("Set: {} - Loss: {}  -  STD: {}".format(epoch, np.mean(lossWindow), std))


def createPretrainedEmbeddings(vocab, tokenizer, vocabFile, vocabSize=5000):
    weights = []
    for w in tokenizer.word_index:
        if (tokenizer.word_index[w] >= vocabSize):
            continue

        if (w not in vocabFile):
            print("Unkown word:", w)
            weights.append(np.random.random(vocab['apple'].shape))
        else:
            weights.append(vocab[w])

    return [tf.convert_to_tensor(weights, dtype=tf.float32)[:, 0]]


if __name__ == '__main__':
    inceptionFiles = Dataset.getTraining3DInceptionFiles()
    validationFile = Dataset.getValidation3DInceptionFiles()
    img2seqs, tokenizer = TextPreProcessing.generateCaptionData()

    model = CaptionModel.CaptionModel(tokenizer, 300, 300, 5000)

    lossFileName = "TrainingData-Simple-Dropout0.pkl"
    modelName = "Simple-Dr0-Weights"
    losses = []
    epoch = 0
    bestValidation, bestEpoch = 100, 0
    while (True):
        for i, f in enumerate(inceptionFiles):
            loss = trainOnInceptionFile(f, img2seqs, model,
                                        epoch="{}/{} E:{}".format(i + 1, len(inceptionFiles), epoch))

            evalLoss = trainOnInceptionFile(validationFile, img2seqs, model,
                                            epoch="Validation".format(i + 1, len(inceptionFiles), epoch),
                                            training=False)

            losses.append({"Epoch": epoch, "File": i, "Train Mean Loss": np.mean(loss), "Train Std Loss": np.std(loss),
                           "Eval Mean Loss": np.mean(evalLoss), "Eval Std Loss": np.std(evalLoss)})

            with open(lossFileName, 'wb') as fp:
                pickle.dump(losses, fp)

            evalLoss = np.mean(evalLoss)
            if (evalLoss < bestValidation):
                model.save_weights(modelName)
                bestValidation = evalLoss
                bestEpoch = epoch
                print("New Best, Saving Model")
            else:
                print("Got validation score: {}, prev best was: {}, at epoch: {}".format(evalLoss, bestValidation,
                                                                                         bestEpoch))
        epoch += 1
