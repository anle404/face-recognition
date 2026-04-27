import tensorflow as tf
from tensorflow.keras import layers

IMAGE_SHAPE=(64, 64, 3)

class DistanceLayer(layers.Layer):
    # Use Squared Euclidean Distance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        an_pos_dist = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_neg_dist = tf.reduce_sum(tf.square(anchor - negative), -1)

        an_pos_dist = tf.expand_dims(an_pos_dist, axis=-1)
        an_neg_dist = tf.expand_dims(an_neg_dist, axis=-1)

        return tf.concat([an_pos_dist, an_neg_dist], axis=1)
    

def get_encoder():
    encoder = tf.keras.Sequential([

        tf.keras.Input(shape=IMAGE_SHAPE),
        layers.Rescaling(1./255.),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2), strides=(2,2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2), strides=(2,2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    
    ], name = "Encoder")
    return encoder

def get_siamese_model(encoder):
    anchor_input   = layers.Input(shape=IMAGE_SHAPE, name="Anchor_Input")
    positive_input = layers.Input(shape=IMAGE_SHAPE, name="Positive_Input")
    negative_input = layers.Input(shape=IMAGE_SHAPE, name="Negative_Input")

    embedded_an = encoder(anchor_input)
    embedded_pos = encoder(positive_input)
    embedded_neg = encoder(negative_input)

    distances = DistanceLayer()(
        embedded_an,
        embedded_pos,
        embedded_neg
    )

    siamese_model = tf.keras.Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )

    return siamese_model

def get_pt_encoder():
    encoder = get_encoder()
    siamese_model = get_siamese_model(encoder)
    siamese_model.load_weights('./verification_weights.h5')

    return encoder

def calculate_distance(emb1, emb2):
    return tf.reduce_sum(tf.square(emb1 - emb2), -1)
