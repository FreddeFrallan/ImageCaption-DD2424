from tensorflow.keras import Model, layers
import tensorflow as tf
from MS_Coco import Dataset
import pickle


def _createModel():
    inLayer1 = layers.Input((8, 8, 2048))
    imgFeature = layers.GlobalAveragePooling2D()(inLayer1)
    return Model(inLayer1, imgFeature)


def flatten3DFile(model, file, outName, batchSize=128):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)

    results = []
    for batchKeys in [list(data.keys())[i:i + batchSize] for i in range(0, len(data), batchSize)]:
        batchRes = model.predict(tf.convert_to_tensor([data[k] for k in batchKeys]))
        results.append(batchRes)

    finalResults = {k: r for k, r in zip(data.keys(), results)}
    with open(outName, 'wb') as fp:
        pickle.dump(finalResults, fp)


def main():
    model = _createModel()
    model.summary()

    for i, f in enumerate(Dataset.getAll3DInceptionFiles()):
        print(i, f)
        flatten3DFile(model, f, "MS-Coco-Train-Flat-{}.pkl".format(i))
