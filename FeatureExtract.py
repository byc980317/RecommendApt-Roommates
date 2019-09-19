import numpy as np
import pandas as pd

# This function does the encoding with count, min, max, std, mean and median of the continuous features
def get_stats(train_df, test_df, target_column, group_column = 'manager_id'):
    '''
    target_column: numeric columns to group with (e.g. price, bedrooms, bathrooms)
    group_column: categorical columns to group on (e.g. manager_id, building_id)
    '''

    # add row id and training samples
    train_df['row_id'] = range(train_df.shape[0])
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0

    # combine train and test df
    all_df = train_df[['row_id', 'train', target_column, group_column]].append(test_df[['row_id','train',
                                                                                        target_column, group_column]])
    #print(all_df.index)
    #print(all_df.loc[:,'row_id'])
    # group column and merge stats (size, mean, std, median)
    grouped = all_df[[target_column, group_column]].groupby(group_column)
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size' % target_column]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean' % target_column]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std' % target_column]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median' % target_column]
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)

    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max' % target_column]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min' % target_column]

    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)

    # the df that contains all stats
    all_df = pd.merge(all_df, the_stats)

    selected_train = all_df[all_df['train'] == 1]
    selected_test = all_df[all_df['train'] == 0]
    # print(selected_train.index)
    # print(selected_train.loc[:,'row_id'])
    selected_train.sort_values('row_id', inplace=True)
    selected_test.sort_values('row_id', inplace=True)
    selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)

    return np.array(selected_train), np.array(selected_test)