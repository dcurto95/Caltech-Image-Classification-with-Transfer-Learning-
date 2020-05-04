import sys

import numpy as np
from keras.applications.resnet50 import preprocess_input
from sklearn.utils import class_weight

import cnn
import data
import plot

if __name__ == '__main__':
    train_generator, valid_generator, test_generator = data.get_image_generators(preprocess_input)

    num_classes = train_generator.num_classes  # Counting BACKGROUND Class
    input_shape = (300, 200, 3)
    print("Test: " + sys.argv[1])
    epochs = 100
    values, counts = np.unique(train_generator.labels, return_counts=True)

    plot.plot_bar(values, counts)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_generator.labels),
                                                      train_generator.labels)
    class_weights = class_weights / class_weights.max()

    class_weights_dict = {}

    for num in np.unique(train_generator.labels):
        class_weights_dict[num] = class_weights[num]

    model = cnn.get_resnet_model(num_classes, input_shape)
    # model = cnn.get_xception_model(num_classes, input_shape)

    cnn.compile_cnn(model)

    history = cnn.fit_generator(model, train_generator, valid_generator, epochs, class_weights_dict)
    plot.draw_history(history, sys.argv[1])

    score = cnn.evaluate_generator(model, test_generator)

    print('\nTest loss: ' + str(score[0]))
    print('Test accuracy: ' + str(score[1]))
