import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y, title=None):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    step = 0.02
    xx = np.arange(x_min, x_max, step)
    yy = np.arange(y_min, y_max, step)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    Z = classifier.predict(grid).reshape(XX.shape)

    plt.figure()
    plt.contourf(XX, YY, Z, alpha=0.35, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='k',
                linewidths=1, cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    xt0, xt1 = int(np.floor(X[:, 0].min() - 1)), int(np.ceil(X[:, 0].max() + 1))
    yt0, yt1 = int(np.floor(X[:, 1].min() - 1)), int(np.ceil(X[:, 1].max() + 1))
    plt.xticks(np.arange(xt0, xt1, 1.0))
    plt.yticks(np.arange(yt0, yt1, 1.0))

    if title is not None:
        plt.title(str(title))

    plt.show()
