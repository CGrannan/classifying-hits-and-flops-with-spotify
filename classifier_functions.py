import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, plot_roc_curve

def clf_fit(clfs, names, xtrain, xtest, ytrain, ytest):
    '''
    Fits a list of classifiers and returns a dataframe of accuracy scores.
    
    Parameters:
    clfs - A list of classifiers.
    names - A list of the names of the classifiers.
    xtrain - A dataframe of train data.
    xtest - A dataframe of test data.
    ytrain - An array of label terms for train data.
    ytest - An array of label terms for test data.
    
    Returns:
    df - A dataframe of each classifier and its train and test accuracy
    '''
    scores = []
    for i, clf in enumerate(clfs):
        name = names[i]
        clf.fit(xtrain, ytrain)
        train_score = clf.score(xtrain, ytrain)
        test_score = clf.score(xtest, ytest)
        scores.append((name, train_score, test_score))
    df = pd.DataFrame(scores, columns=['classifier', 'train_score', 'test_score'])
    return df

def grid_fit(clf, params, xtrain, ytrain):
    '''
    Fits a GridSearchCV to a classifier and prints the best parameters and best accuracy.
    
    Parameters:
    clf - A classifier instance.
    params - A dictionary containing the parameters to be changed.
    xtrain - A dataframe of train data.
    ytrain - An array of label terms for train data.
    
    Returns:
    gridsearch - A fit GridSearchCV instance.
    '''
    gridsearch = GridSearchCV(clf, params, scoring='accuracy', cv=3, verbose=0)
    gridsearch.fit(xtrain, ytrain)
    print(f"Best params: {gridsearch.best_params_}")
    print(f"Best score: {gridsearch.best_score_}")
    return gridsearch

def classifier_results(clf, xtrain, xtest, ytrain, ytest, train_preds, test_preds):
    '''
    Prints diagnostics for a classifier.
    
    Parameters:
    clf - A classifier.
    xtrain - A dataframe of training data.
    xtest - A dataframe of testing data.
    ytrain - An array of label terms for training data.
    ytest - An array of label terms for testing data.
    train_preds - Predictions for the training data.
    test_preds - Predictions for the testing data.
    
    Returns:
    auc_train - Area under the ROC curve for train data.
    auc-test - Area under the ROC curve for test data.
    '''
    print('Classification Report - Training')
    print(classification_report(ytrain, train_preds))
    print('Classification Report - Testing')
    print(classification_report(ytest, test_preds))
    
    fpr, tpr, thresholds = roc_curve(train_preds, ytrain)
    auc_train = round(auc(fpr, tpr),3)
    
    fpr, tpr, thresholds = roc_curve(test_preds, ytest)
    auc_test = round(auc(fpr, tpr),3)
    
    print(f'Training AUC: {auc_train}, Testing AUC: {auc_test}')
    
    print('Confusion Matrix - Training')
    plot_confusion_matrix(clf, xtrain, ytrain, normalize='true')
    plt.show()
    print('Confusion Matrix - Testing')
    plot_confusion_matrix(clf, xtest, ytest, normalize='true')
    plt.show()
    
    cm = confusion_matrix(ytest, test_preds)
    
    lst = [cm[0,0], cm[0,1], cm[1,1], cm[1,0]]
    labels = ['Identified Flops', 'Misidentified Flops', 'Identified Hits', 'Misidentified Hits']
    
    plt.figure(figsize=(10,6))
    plt.title('Test Results')
    sns.barplot(labels, lst, palette=['#d01c8b', '#f1b6da', '#4dac26', '#b8e186'])
    plt.show()
    
    return auc_train, auc_test