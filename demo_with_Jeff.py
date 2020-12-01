import numpy as np
import pandas as pd

def demo(song_list, df_stats, df_data, scaler, svm_model, forest_model):
    '''
    Tests a set of songs to see if they are hits or flops.
    
    Parameters:
    song_list - List of song indexes to be tested.
    df_stats - dataframe to pull song names from.
    df_data - dataframe of processed data to pull from.
    scaler - fit data scaler to transform test songs.
    svm_model - model to be tested.
    forest_model - model to be tested.
    
    Returns:
    The name of the test songs and whether they are a hit or flop for both models.
    '''
    songs = df_data.iloc[song_list]
    
    dummies = pd.get_dummies(songs['time_signature'], prefix='time_signature')
    cat = pd.concat([dummies, songs['mode']], axis=1)

    cont_var = songs.drop(['time_signature', 'mode'], axis=1)
    scaled = pd.DataFrame(scaler.transform(cont_var), index=cont_var.index, columns = cont_var.columns)

    sample_test = pd.concat([scaled, cat], axis=1)
    
    if 'time_signature_1' not in sample_test.columns:
        sample_test['time_signature_1'] = 0
    if 'time_signature_3' not in sample_test.columns:
        sample_test['time_signature_3'] = 0
    if 'time_signature_4' not in sample_test.columns:
        sample_test['time_signature_4'] = 0
    if 'time_signature_5' not in sample_test.columns:
        sample_test['time_signature_5'] = 0
    if 'time_signature_2' in sample_test.columns:
        sample_test = sample_test.drop(['time_signature_2'], axis=1)
    print('SVM Results:')
    svm_results = svm_model.predict(sample_test)
    for i, result in enumerate(svm_results):
        name = df_stats.iloc[song_list[i]]['track']
        if result == 1:
            hit = 'hit'
        else:
            hit = 'flop'
        print(f'{name}: {hit}')
    print('\n')
    print("Random Forest Results:")
    forest_results = forest_model.predict(sample_test)
    for i, result in enumerate(forest_results):
        name = df_stats.iloc[song_list[i]]['track']
        if result == 1:
            hit = 'hit'
        else:
            hit = 'flop'
        print(f'{name}: {hit}')
    
    return sample_test