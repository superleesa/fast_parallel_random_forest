import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
__author__ = "Satoshi Kashima"


from random_forest import RandomForest as rf1
from parallel_random_forest import RandomForest as rf2
from data_loader import load_files

if __name__ == "__main__":
    rf1_accuracy = []
    rf2_accuracy = []
    rf1_training_time = []
    rf2_training_time = []

    # n_estimators_trials = [10, 20, 30, 40, 50, 60, 80, 90, 100]
    # n_estimators_trials = [10, 20, 30]
    n_files = [1, 10, 30, 50]

    for n in n_files:
        X_train, X_test, y_train, y_test = load_files(n, 20, True)

        # Create an instance of the DecisionTree class
        model1 = rf1(n_estimators=50, max_depth=7, random_state=32678940)
        model2 = rf2(n_estimators=50, max_depth=7, random_state=32678940, n_processes=8)

        # Fit the tree to the training data
        rf1_start_timer = default_timer()
        model1.fit(X_train, y_train)
        rf1_end_time = default_timer()
        rf1_training_time.append(rf1_end_time - rf1_start_timer)

        rf2_start_timer = default_timer()
        model2.fit(X_train, y_train)
        rf2_end_time = default_timer()
        rf2_training_time.append(rf2_end_time - rf2_start_timer)

        # Make predictions on the test set
        y_pred1 = model1.predict(X_test)
        y_pred2 = model2.predict(X_test)


        # Evaluate the accuracy of the model
        accuracy1 = np.mean(y_pred1 == y_test)
        accuracy2 = np.mean(y_pred2 == y_test)
        # print(f"Accuracy: {accuracy}")
        rf1_accuracy.append(accuracy1)
        rf2_accuracy.append(accuracy2)

    # plt.scatter(n_estimators_trials, rf1_accuracy, label="Regular Random Forest")
    # plt.scatter(n_estimators_trials, rf2_accuracy, label="Parallel Random Forest")
    # plt.title("Accuracy of Regular RF vs. PRF")
    # plt.xlabel("Number of trees (n_estimators)")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="lower right")
    # plt.show()

    plt.scatter(n_files, rf1_training_time, label="Regular Random Forest")
    plt.scatter(n_files, rf2_training_time, label="Parallel Random Forest")
    plt.title("Training Time of Regular RF vs. PRF")
    plt.xlabel("Number of Files (data size)")
    plt.ylabel("Training Time (s)")
    plt.legend(loc="lower right")
    plt.show()