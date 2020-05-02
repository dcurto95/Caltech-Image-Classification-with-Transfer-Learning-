import itertools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid import axes_size
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_history(history, test_name):
    # list all data in history
    print(history.history.keys())

    plt.figure()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig('../logs/' + test_name + '/model_acc.jpg')

    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig('../logs/' + test_name + '/model_loss.jpg')

    plt.close('all')


def plot_bar(x, y):
    plt.figure()
    plt.bar(x, y)
    plt.show()


def plot_confusion_matrix(
        cm,
        classes,
        predicted_classes,
        test_name,
        normalize=False,
        title='Confusion matrix',
        file_title='file',
        folder_title="",
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, predicted_classes, rotation="vertical")
    plt.yticks(tick_marks, classes)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(
        ("../logs/" + test_name + "/confusion.png"))
    plt.close()
