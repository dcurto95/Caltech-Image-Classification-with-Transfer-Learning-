from keras import applications, Model, optimizers
from keras.layers import Dense, Flatten


def get_resnet_model(num_classes, input_shape):
    model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    # for layer in model.layers[:5]:
    #     layer.trainable = False
    #
    # # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model
    model_final = Model(inputs=model.input, output=predictions)
    model_final.summary()
    return model_final


def get_xception_model(num_classes, input_shape):
    model = applications.Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    # # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    # for layer in model.layers[:5]:
    #     layer.trainable = False
    #
    # # Adding custom Layers
    x = model.output
    # x = Flatten()(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model
    model_final = Model(inputs=model.input, output=predictions)
    model_final.summary()
    return model_final


def compile_cnn(model):
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=["accuracy"])


def fit_generator(model, train_generator, valid_generator, epochs, class_weights):
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    if not class_weights:
        return model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=1,
            validation_data=valid_generator)

    return model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        validation_data=valid_generator,
        class_weight=class_weights)


def predict_generator(model, test_generator):
    test_steps = test_generator.samples // test_generator.batch_size

    return model.predict_generator(test_generator, steps=test_steps)


def evaluate_generator(model, test_generator):
    return model.evaluate_generator(test_generator, steps=test_generator.samples // test_generator.batch_size,
                                    verbose=0)
