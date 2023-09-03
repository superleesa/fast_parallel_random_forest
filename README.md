# Fast Paralll Random Forest
In this project, I implemented a fast parallel Random Forest, according to a published research paper by J. Chen et al, [A Parallel Random Forest Algorithm for Big Data in a Spark Cloud Computing Environment](https://arxiv.org/abs/1810.07748).

## What's implemented specifically?
I implemented two versions of Random Forest:
1. Regular Random Forest with decision trees, bagging, and a random feature selection.
2. Improved Random Forest Algorithm from the paper, including dimension reduction, weighted voting, and parallel training of tree estimators.

## Note on running the code
Before running the code, please ensure that you have the following dependencies installed.
- Python 3.9
- Numpy 1.21.5
- Scikit-learn 1.02
- matplotlib
- os (built-in)
- multiprocessing (built-in)
- timeit (build-in)

On the command prompt run: “python main2.py” to run the RF and PRF on wine dataset from scikit learn. This will train both models and create a comparison plot in terms of execution time.

On the command prompt run: “python main.py” to run the RF and PRF on URL Reputation dataset. This plots the execution time after finishing both training and testing.  To run this command, the URL Reputation dataset (https://archive.ics.uci.edu/ml/datasets/URL+Reputation) must be downloaded and unzipped to the directory inside the package. Also, the directory location to the dataset in data_loader.py might need to be changed depending on the OS.
Training on the whole dataset of URL Reputation took more than 1 hour on our computer. I suggest using smaller data such as Iris data or wine data from scikit-learn. 

## Detailed Note on Implementation
### Variable Importance
In fit function of parallel_random_forest.py, I calculated the entropy, split information, and gain information (I did not include the gain ratio since it was computationally expensive). Then, we calculated the variable importance of each feature according to the gain information (the bigger the gain information, the bigger the variable importance). 
According to the paper, we select the top k important features and other choices of remaining variables are random. Hence, we randomly select the raming features using np.random.randint and combined the important features and the randomly chosen features.

### Weighted Voting
The last portion of the fit function computes the weights of each tree estimator using OOB data. It traverses through each estimator and calculates the number of correctly predicted samples over all samples with the equation: weight = np.sum(predictions == y_oob) / len(y_oob)

The last portion of the predict function uses these precomputed weights to weigh each prediction made by the estimator. It sums up the weight of votes for each class and returns the class with the maximum total weight.

### Task Parallelism
I modularized the samples/features selection process and the decision tree creation process to create several estimators in parallel. 

This function is mapped to each processor using the multiprocessing library. I used pool.apply_async function to parallelize the training process (in the original paper, author used Spark since they were working on a distributed environment). Given a task to train n trees, we split this task into n/num_processes tasks and each process takes a responsibility train part of the ensemble model.


