import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


def plot_history(history,
                 path=None,
                 figsize=(10, 4), 
                 is_sparse=False):
    txt = 'sparse_' if is_sparse else ''
    
    plt.subplots(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(history.history[f'{txt}categorical_accuracy'])
    plt.plot(history.history[f'val_{txt}categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True, figure_path=None,
                          cmap=plt.cm.Blues,
                          figsize=(7, 7)):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True labels',
        xlabel='Predicted labels'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylim(len(classes) - 0.5, -0.5)

    fig.tight_layout()
    # plt.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches='tight')
    plt.show()


def plot_segment(seq, figsize=None, vmin=None, vmax=None, titles=None):
    plt.subplots(figsize=figsize)
    plt.subplots_adjust(hspace=1)

    for i, s in enumerate(seq):
        s = s.reshape([1, -1])
        ax = plt.subplot(len(seq), 1, i + 1)
        ax.title.set_text(titles[i])
        plt.pcolor(s, vmin=vmin, vmax=vmax)

    plt.show()

