# Predicting Hits and Flops with Spotify

For this project I am working with information from spotify. The data includes different descriptive statistics of tracks and seeks to classify those tracks as hits or flops. The target is a binary variable indicating if a song was a hit. For a full overview of the column meanings, there is a readme file inside the archive folder. This file goes over all of the variables as well as defines the criteria for a song being labeled a hit. The end goal will be to see if a recording label is able to indicate whether a song is likely to comercially succeed before being officially released. Here is a brief outline of the project, but for a more indepth look make sure to look at the notebook.ipynb file.

## Data Preprocessing & EDA

We made a few interesting decisions during data cleaning. We removed two instances where there was no time signature for a track, and we removed a few outliers from the duration_ms, chorus_hit, and sections columns. After removing these outliers we still had over 31,000 tracks. At this point, we explored some relationships within the data. First we looked at the distribution of hits by decade.

![images/hits_by_decade.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/hits_by_decade.png)

It seems like we have many more hits represented in the earlier decades. We also checked some correlations between variables. We had two strong negative correlations and two strong positive correlations. First, here are the negative correlations:

![images/accousticness_energy.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/accousticness_energy.png)

![images/accousticness_loudness.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/accousticness_loudness.png)

We can see that 'accousticness' correlates negatively with both energy and loudness. This is understandable as acoustic music usually tends to be softer and quieter compared to music with electric imstruments. Here are the two positive correlations:

![images/loudness_energy.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/loudness_energy.png)

![images/duration_sections.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/duration_sections.png)

Here we see two more relationships that make sense. Our definition of energy includes loudness, so naturally they would correlate. Similarly, a longer song is expected to have more sections. At this point we were ready to start building classifiers.

## Modeling

We decided to use a variety of classifiers for this project, and fine-tune each of them as much as possible before selecting our final model. We used logistic regression, k-nearest neighbors, support vector machine, random forest, and adaboost models. To begin modeling, we established a baseline for each of these types of classifiers by using the default parameters. Of these models, the random forest had the best accuracy at 79.9%. To take this further we attempted to use principal component analysis to reduce dimensionality. When we ran the models on the PCA transformed data, all models produced worse results. At this point we decided to drop the PCA approach. Next, we used GridSearchCV to fine-tune the parameters for each model to get the best result we could. Once we had the best possible model for each approach, we inspected the confusion matrixes and areas under the ROC curve. Here are the resulting matrices for each model.

Logistic Regression:

![images/logreg/matrix.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/logreg_matrix.png)

K-nearest neighbors:

![images/knn_matrix.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/knn_matrix.png)

Support vector machine:

![images/svm_matrix.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/svm_matrix.png)

Random forest:

![images/forest_matrix.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/forest_matrix.png)

Adaboost:

![images/ada_matrix.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/ada_matrix.png)

All models have good accuracy and precission, but they also all show a tendency to mislabel flops as hits resulting in a relatively high false positive rate. The random forest model has the lowest false positive rate, but even still shows this tendency.

And here is the training and testing ROC curves plotted:

![images/train_roc.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/train_roc.png)

![images/test_roc.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/test_roc.png)

The models all perform well, but with higher false positive rates than I would like. The SVM model is slightly more accurate with an AUC of .80, but the random forest model has a lower false positive rate. Ultimately, I think the SVM model is the better model unless we are looking to be conservative, in which case the random forest is better. Finally, we used permutation importance to determine the weights of our features. Here are the weights for our two best models:

SVM features:

![images/svm_features.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/svm_features.png)

Random forest features:

![images/forest_features.png](https://github.com/CGrannan/dsc-mod-5-project-online-ds-sp-000/blob/master/images/forest_features.png)

The two models have similar orders for features, especially at the top of the list, but ascribe fairly different weights. We should note that 'instrumentalness' and 'accousticness' are the two best features for both models.

## Summary

We built two models that we can use to predict if a song will be a commercial hit. Our SVM model has a higher overall accuracy, correctly classifying songs about 80% of the time. However, if we are trying to conservatively predict flops so as to avoid risky investments, then we should be using the random forest classifier. This model had the lowest rate of false positives, meaning we have a lower chance of investing in marketing campaigns that are destined to fail. Further more, when looking at what makes a commercially successful song, we were able to distinguish key features used by both models. Instrumentalness is key it seems as both models placed that as the strongest feature. Accousticness, danceability, energy and duration round out the rest of the top 5 in both models, though not in the same order. When deciding on which tracks to support, we should be keeping these key factors in mind.

## Future Work

There are a few ways that I think we can expand on this project. First, we can narrow the focus of our predictions down to just classifying tracks released in the last 5 or 10 years. This will allow us to focus on the market demands of the current generation, rather than looking at overarching trends over the last several decades. Second, I would look at testing this model by genre of music. Will we have similar results when comparing country music and opera music? Perhaps we can get a better accuracy if we train and use our model on individual genres, rather than all tracks.