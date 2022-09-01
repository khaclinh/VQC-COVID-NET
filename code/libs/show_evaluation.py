import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import logging


def plot_confusion_matrix(y_true, y_pred,
                          labels=None,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          figsize=None,
                          save_path=None):
    """
    :param y_true: 1d array containing y true
    :param y_pred: 1d array containing y pred
    :param labels: 1d array containing label names
    :param normalize: whether to normalize confusion matrix to 0-1
    :param cmap: plt color
    :param figsize: figure size
    :param save_path: path to save figure
    """

    # Set default label names
    if labels is None:
        labels = np.unique(y_true)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Select only existing labels from y true and y pred
    unique_labels = np.unique(np.array([y_true, y_pred])).astype(np.int)
    labels = labels[unique_labels]

    # Normalize confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    _ = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Show all ticks
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True labels',
        xlabel='Predicted labels'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm[np.where(~np.isnan(cm))].max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Adjust layout
    plt.ylim(len(labels) - 0.5, -0.5)
    fig.tight_layout()
    plt.tight_layout()

    # Save figure to file
    if save_path is not None:
        plt.savefig(save_path)
        logging.info('figure saved')

    plt.show()


def show_classification_report(y_true, y_pred, labels=None, digits=4,
                               save_path=None):
    """
    :param y_true: 1d array containing y true
    :param y_pred: 1d array containing y pred
    :param digits: number of digits to show
    :param labels: list label names
    :param save_path: path to save report
    """

    # Select only existing labels from y true and y pred
    unique_labels = np.unique(np.array([y_true, y_pred])).astype(np.int)
    labels = labels[unique_labels]

    # show report
    report = classification_report(y_true, y_pred, digits=digits,
                                   target_names=labels)
    logging.info(report)

    # save to file
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(report)


def plot_keras_training_curve(history, figsize=None, is_sparse=True,
                              save_path=None):
    """
    :param history: history dictionary, returned by Keras fit() method
    :param figsize: figure size
    :param is_sparse: whether the model use sparse loss
    :param save_path: path to save figure
    """

    # plot loss curve
    plt.subplots(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title('loss')
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.legend()

    # plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.title('accuracy')
    sparse = 'sparse_' if is_sparse else ''
    plt.plot(history[f'{sparse}categorical_accuracy'], label='train')
    plt.plot(history[f'val_{sparse}categorical_accuracy'], label='valid')
    plt.legend()
    plt.tight_layout()

    # save to file
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
