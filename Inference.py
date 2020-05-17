import TextPreProcessing
import tensorflow as tf
import Model, Utils
import numpy as np


def generateImgFeatures(featureModel, imgPaths):
    temp_input = tf.convert_to_tensor([Utils.load_image(p) for p in imgPaths])
    return featureModel(temp_input)


def _beamSearchImage(model, features, tokenizer, maxLen, beamSize=3):
    dec_input = tf.convert_to_tensor([tokenizer.word_index['<start>'] for _ in range(beamSize)])
    hidden = model.decoder.reset_state(batch_size=beamSize)
    features = tf.broadcast_to(features, (beamSize,) + features.shape)

    results, propScores = [], [0] * beamSize
    for i in range(maxLen):
        dec_input = tf.expand_dims(dec_input, axis=1)
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)
        if (i == 0):
            predicted_id = np.argsort(predictions[0])[-beamSize:]
        else:
            predicted_id = np.argmax(predictions, axis=-1)

        for i, pred in enumerate(predicted_id):
            propScores[i] += np.log(predictions[i][pred].numpy())

        results.append([tokenizer.index_word[w] for w in predicted_id])
        dec_input = tf.convert_to_tensor(predicted_id)

    results = [(t, p) for t, p in zip(Utils.cleanseOutputs(results), propScores)]
    return sorted(results, key=lambda x: x[1], reverse=True)


def beamSearchTrue(model, imgFeatures, tokenizer, maxLen, beamSize=3):
    dec_input = tf.convert_to_tensor([tokenizer.word_index['<start>'] for _ in range(beamSize)])
    hidden = model.decoder.reset_state(batch_size=beamSize)
    features = tf.broadcast_to(imgFeatures, (beamSize,) + imgFeatures.shape)

    # Beams = (Prob, hidden, history)
    beams = [[(0, None, [])] for i in range(beamSize)]
    for i in range(maxLen):
        dec_input = tf.expand_dims(dec_input, axis=1)
        #print("Dec Input:", dec_input.shape)
        #print("Hidden:", hidden.shape)
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)
        predictions = tf.math.softmax(predictions, axis=1)
        if (i == 0):
            beams = [(np.log(predictions[0][i]), hidden[0], [i]) for i in np.argsort(predictions[0].numpy())[-beamSize:]]
        else:
            allBeams = []
            for j, pred in enumerate(predictions.numpy()):
                prevProb, _, prevHistory = beams[j]
                topPerformers = np.argsort(pred)[-beamSize:]
                # for k in topPerformers:
                # allBeams.append((prevProb + pred[k], hidden[k]))
                allBeams.extend([(prevProb + np.log(pred[k]), hidden[j], prevHistory + [k]) for k in topPerformers])

            allBeams.sort(key=lambda x: np.exp(x[0]), reverse=True)
            beams = allBeams[:beamSize]

        '''
        print("Beams:", len(beams))
        for b in beams:
            print(b[0], b[1].shape, b[2])
        '''

        hidden = tf.concat([tf.expand_dims(b[1], 0) for b in beams], axis=0)
        dec_input = tf.convert_to_tensor([b[2][-1] for b in beams])

    results = [(prob, Utils.cleanseSent([tokenizer.index_word[w] for w in history])) for prob, _, history in beams]
    return sorted(results, key=lambda x: x[0], reverse=True)


def beamSearchImages(model, featureModel, tokenizer, imgPaths, maxLen):
    features = generateImgFeatures(featureModel, imgPaths)
    results = [beamSearchTrue(model, f, tokenizer, maxLen, beamSize=20) for f in features]
    return results


def greedyInferenceWithFeatures(model, features, tokenizer, maxLen):
    dec_input = tf.convert_to_tensor([tokenizer.word_index['<start>'] for _ in features])
    hidden = model.decoder.reset_state(batch_size=len(features))

    results = []
    for i in range(maxLen):
        dec_input = tf.expand_dims(dec_input, axis=1)
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)

        predicted_id = np.argmax(predictions, axis=-1)
        results.append([tokenizer.index_word[w] for w in predicted_id])
        dec_input = tf.convert_to_tensor(predicted_id)

    return Utils.cleanseOutputs(results)


def greedyInference(model, featureModel, tokenizer, imgPaths, maxLen):
    features = generateImgFeatures(featureModel, imgPaths)
    return model.greedyInference(features, maxLen, tokenizer)


if __name__ == '__main__':
    img2seqs, tokenizer = TextPreProcessing.generateCaptionData()

    print("Loading Model")
    model = Model.CaptionModel(tokenizer, 300, 600, 5000, useFullImgEncoder=True)
    # model.load_weights("Attention-GRU-Dr0.15-ImageCaptionModel-MultiModalWordEmb-300-Weights")
    model.load_weights("Attention-Dr0.2-ImageCaptionModel-Weights")
    '''
    model = SimpleModel.CaptionModel(tokenizer, 512, 512, 5000)
    model.load_weights('LSTM-ImageCaptionModel-Weights')
    '''
    featureModel = Utils.generateInceptionFeatureModel()

    imgPaths = ["Self Imposed Isolation.png", "Petra.jpg", "Mange.jpg", "Träd Dude.png", "cars.jpg", "lions.jpg"]
    #imgPaths = ["cat.jpg", "cars.jpg", "fotball.jpg", "fat.jpg", "dammsugare.png", "elephant.jpeg"]
    # imgPaths = ['Joey.png', 'Joey2.png', 'Adrian1.png', 'Adrian2.png', 'Joey3.png']
    features = generateImgFeatures(featureModel, imgPaths)
    results = model.greedyInference(features, 20, tokenizer)
    for r in results:
        print(r)

    fModel = Model.FullInceptionEncoder(model.encoder, model.imgEncoder)
    results = beamSearchImages(model, fModel, tokenizer, imgPaths, 20)
    for r in results:
        print(r[0])
    '''
    '''

    '''
    # for i in range(10):
    res = greedyInference(model, featureModel, tokenizer, imgPaths, 20)
    print(res)
    res2 = beamSearchImages(model, featureModel, tokenizer, imgPaths, 20)
    print(res2)
    '''
