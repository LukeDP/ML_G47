import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve

class ClassificationUtils:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name):
        """
        Plot the confusion matrix for the given model.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.show()

    @staticmethod
    def plot_learning_curve(estimator, X, y, title, cv=5, scoring="accuracy"):
        """
        Plot the learning curve for the given estimator.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel(scoring.capitalize())
        plt.legend(loc="best")
        plt.grid()
        plt.show()
