import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    'bert_en_uncased_L-12_H-768_A-12_3':
        '/home/ubuntu/models/encoder/bert_en_uncased_L-12_H-768_A-12_3',
    'bert_en_wwm_uncased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/encoder/bert_en_wwm_uncased_L-24_H-1024_A-16_3',
    'bert_en_wwm_cased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/encoder/bert_en_wwm_cased_L-24_H-1024_A-16_3',
    'bert_en_cased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/encoder/bert_en_cased_L-24_H-1024_A-16_3',
    'experts_bert_wiki_books_2':
        '/home/ubuntu/models/encoder/experts_bert_wiki_books_2',
    'experts_bert_wiki_books_mnli_2':
        '/home/ubuntu/models/encoder/experts_bert_wiki_books_mnli_2',
    'bert_en_uncased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/encoder/bert_en_uncased_L-24_H-1024_A-16_3',
    'bert_multi_cased_L-12_H-768_A-12_3':
        '/home/ubuntu/models/encoder/bert_multi_cased_L-12_H-768_A-12_3',
    'albert_en_base_2':
        '/home/ubuntu/models/encoder/albert_en_base_2',
    'albert_en_large_2':
        '/home/ubuntu/models/encoder/albert_en_large_2',
    'albert_en_xlarge_2':
        '/home/ubuntu/models/encoder/albert_en_xlarge_2',
    'albert_en_xxlarge_2':
        '/home/ubuntu/models/encoder/albert_en_xxlarge_2',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_uncased_L-12_H-768_A-12_3':
        '/home/ubuntu/models/preprocess/bert_en_uncased_preprocess_3',
    'bert_en_wwm_uncased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/preprocess/bert_en_uncased_preprocess_3',
    'bert_en_wwm_cased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/preprocess/bert_en_cased_preprocess_3',
    'bert_en_cased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/preprocess/bert_en_cased_preprocess_3',
    'experts_bert_wiki_books_2':
        '/home/ubuntu/models/preprocess/bert_en_uncased_preprocess_3',
    'experts_bert_wiki_books_mnli_2':
        '/home/ubuntu/models/preprocess/bert_en_uncased_preprocess_3',
    'bert_en_uncased_L-24_H-1024_A-16_3':
        '/home/ubuntu/models/preprocess/bert_en_uncased_preprocess_3',
    'bert_multi_cased_L-12_H-768_A-12_3':
        '/home/ubuntu/models/preprocess/bert_multi_cased_preprocess_3',
    'albert_en_base_2':
        '/home/ubuntu/models/preprocess/albert_en_preprocess_2',
    'albert_en_large_2':
        '/home/ubuntu/models/preprocess/albert_en_preprocess_2',
    'albert_en_xlarge_2':
        '/home/ubuntu/models/preprocess/albert_en_preprocess_2',
    'albert_en_xxlarge_2':
        '/home/ubuntu/models/preprocess/albert_en_preprocess_2',
}

BERT_MODEL_NAME = 'bert_en_wwm_uncased_L-24_H-1024_A-16_3'
tfhub_handle_encoder = map_name_to_handle[BERT_MODEL_NAME]
tfhub_handle_preprocess = map_model_to_preprocess[BERT_MODEL_NAME]

BATCH_SIZE = 8
TRAIN_VAL_SPLIT_RATIO = 0.2
SEED = 123
checkpoint_filepath = 'ckpt'


def load_train_data():
    cleaned_dataset_dir = 'cleaned_data'
    buffer_size = BATCH_SIZE * 2
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        directory=cleaned_dataset_dir,
        label_mode='int',
        batch_size=BATCH_SIZE,
        validation_split=TRAIN_VAL_SPLIT_RATIO,
        subset='training',
        seed=SEED)
    train_ds = raw_train_ds.cache().shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        directory=cleaned_dataset_dir,
        label_mode='int',
        batch_size=BATCH_SIZE,
        validation_split=TRAIN_VAL_SPLIT_RATIO,
        subset='validation',
        seed=SEED)
    val_ds = val_ds.cache().shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    class_names = raw_train_ds.class_names
    for text_batch, label_batch in train_ds.take(1):
        for i in range(6):
            print(f'Review: {text_batch.numpy()[i]}')
            label = label_batch.numpy()[i]
            print(f'Label : {label} ({class_names[label]})')

    return train_ds, val_ds


def build_classifier_model(use_lstm=False):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    if not use_lstm:
        net = outputs['pooled_output']
    else:
        net = outputs['sequence_output']
        net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2048), name="bilstm")(net)
    # net = tf.keras.layers.Dense(128, activation='relu', name="relu")(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


def train_bert(epochs, init_lr, use_lstm=False, load_ckpt=False):
    train_ds, val_ds = load_train_data()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    steps_per_epoch = len(train_ds)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    classifier_model = build_classifier_model(use_lstm=use_lstm)
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)
    if load_ckpt:
        if use_lstm:
            classifier_model.load_weights(checkpoint_filepath + "_lstm")
        else:
            classifier_model.load_weights(checkpoint_filepath)

    if use_lstm:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath + '_lstm',
            save_weights_only=True,
            monitor='val_binary_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    else:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_binary_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    loss, acc = classifier_model.evaluate(val_ds, verbose=1)
    print("Before training, accuracy: {:5.2f}%".format(100 * acc))
    classifier_model.fit(
        x=train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback],
    )
