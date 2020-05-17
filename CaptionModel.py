from tensorflow.keras import Model, layers
import tensorflow as tf
import Utils


# Final transformation that takes the LSTM output and gives us a distribution over the vocab
def _createVocabSelection(embSize=512, vocabSize=5000):
    inLayer = layers.Input(embSize)
    vocabSelection = layers.Dense(vocabSize, 'softmax')(inLayer)
    return Model(inLayer, vocabSelection)


# Create the linear transformation layer between the image feature vector and the LSTM
def _createFeatureEncoder(embSize):
    encodeLayer = layers.Dense(embSize, 'linear')
    inLayer = layers.Input((8, 8, 2048))
    imgFeature = layers.GlobalAveragePooling2D()(inLayer)
    #inLayer = layers.Input(2048)
    f2 = encodeLayer(imgFeature)
    return Model(inLayer, f2)


class CaptionModel(tf.keras.Model):

    def __init__(self, tokenizer, embeddingDim, units, vocabSize, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

        self.embeddings = tf.keras.layers.Embedding(vocabSize, embeddingDim)
        self.decoder = LSTM_Decoder(embeddingDim, units, vocabSize)
        self.imgEncoder = _createFeatureEncoder(embeddingDim)
        self.imgEncoder.summary()

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def greedyInference(self, features, maxLen, tokenizer=None):
        seqOutputs = []

        hidden = self.decoder.generateStartState(batch_size=features.shape[0])
        features = self.imgEncoder(features)
        pred, hidden = self.decoder(features, hidden)
        out = tf.argmax(pred, axis=-1)
        seqOutputs.append(out.numpy())

        for i in range(1, maxLen):
            pred, hidden = self._textInferenceStep(out, hidden, False)
            out = tf.argmax(pred, axis=-1)
            seqOutputs.append(out.numpy())

        if (tokenizer == None):
            return seqOutputs
        return Utils.cleanseOutputs([[tokenizer.index_word[w] for w in turn] for turn in seqOutputs])

    @tf.function
    def train_step(self, img_tensor, target, training=True):
        batchSize, seqLen, loss = target.shape[0], target.shape[1], 0
        hidden = self.decoder.generateStartState(batch_size=batchSize)

        with tf.GradientTape() as tape:
            features = self.imgEncoder(img_tensor)

            # Pass image
            predictions, hidden = self.decoder(features, hidden, training)
            loss += self.loss_function(target[:, 0], predictions)

            # Pass text
            for i in range(seqLen - 1):
                predictions, hidden = self._textInferenceStep(target[:, i], hidden, training)
                seqLoss = self.loss_function(target[:, i + 1], predictions)
                loss += seqLoss

        avgLoss = loss / seqLen
        if (training):
            trainable_variables = self.imgEncoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, avgLoss

    # Given word indicies extract the relevant word embeddings and pass it to the LSTM
    def _textInferenceStep(self, textIn, hidden, training):
        embIn = self.embeddings(textIn)
        return self.decoder(embIn, hidden, training)


class LSTM_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(LSTM_Decoder, self).__init__()

        self.units = units
        self.lstm = tf.keras.layers.LSTMCell(self.units, recurrent_initializer='glorot_uniform')

        self.vocabSelection = _createVocabSelection(embedding_dim, vocab_size)
        self.vocabSelection.summary()

    def call(self, x, hidden, training=False):
        # print("X", x.shape, " Hidd:", hidden[0].shape, hidden[1].shape)
        output, states = self.lstm(x, hidden, training)
        return self.vocabSelection(output), states

    def generateStartState(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
