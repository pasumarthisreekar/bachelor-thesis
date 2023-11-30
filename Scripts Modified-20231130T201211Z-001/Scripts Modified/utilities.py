from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from matplotlib import cm
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_data(base_dir, file_name, header=True, y=None, dtype=np.float32):
    path = join(base_dir, file_name)
    df = pd.read_csv(f'{path}.csv', header=header)
#     print(df.head())
    if y is None:
        return df.to_numpy(dtype=dtype)
    else:
        return df.iloc[:, :-y].to_numpy(dtype=dtype), df.iloc[:, -y:].to_numpy(dtype=dtype)

    
def transform_9D(array):
    x, y = array[:, 0], array[:, 1]
    dim1 = x + y
    dim2 = x - y
    dim3 = x * y
    dim4 = x * x
    dim5 = y * y
    dim6 = dim3 * x
    dim7 = dim3 * y
    dim8 = dim4 * x
    dim9 = dim5 * y
    
    return np.array([
        dim1, dim2, dim3, dim4,
        dim5, dim6, dim7, dim8,
        dim9
    ]).T
# def split_data(X, y=None, seed=42, test_split=None, dev_split=None):
#     if test_split is None:
#         raise Exception('Train test split not specified')
#     elif test_split > 1 or test < 0:
#         raise Exception('Invalid test split specified')

#     # if dev_split is None and 

#     if y is not None:
#         if dev_split is None:
#             return train_test_split(X, train_size=1-dev_split, random_state=seed)
#         else:
#             X_train, X_dev = train_test_split(X, train_size=1-dev_split, random_state=seed)
#             X_train, X_test = train_test_split(X_train, train_size=1-test_split, random_state=seed)

#     else:
#         if dev_split is not None:
#             return train_test_split(X, y, train_size=1-test_split, random_state=seed)
#         else:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1-dev_split, random_state=seed)
#             X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, train_size=1-test_split, random_state=seed)
#             return X_train, X_dev, X_test, y_train, y_dev, y_test

def scale_data(X, scale, feature_range=None):
    if scale == 'minmax':
        if feature_range is None:
            feature_range = (0, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scale == 'standard':
        scaler = StandardScaler()
    else:
        raise Exception('Invalid scale option specified')
    return scaler.fit_transform(X), scaler

def comp_plot(X_orig, X_recons, X_latent, y, title):

    fig = plt.figure(figsize=(30, 15))
    fig.suptitle(title, fontsize=16)

#     ax1 = fig.add_subplot(1, 2, 1, projection='3d', title='Original Distribution')
#     ax1.scatter(X_orig[:,0], X_orig[:,1], X_orig[:,2], marker='o', c=y, cmap=cm.get_cmap('plasma'))

    ax2 = fig.add_subplot(1, 2, 1, projection='3d', title='Reconstructed Distribution')
    ax2.scatter(X_recons[:,0], X_recons[:,1], X_recons[:,2], marker='o', c=y, cmap=cm.get_cmap('plasma'))
    ax2.set_xlim3d(left=0.0, right=1.0)
    ax2.set_ylim3d(bottom=0.0, top=1.0) 
    ax2.set_zlim3d(bottom=0.0, top=1.0)
    
    ax3 = fig.add_subplot(1, 2, 2, title='Latent Space')
    ax3.scatter(X_latent[:,0], X_latent[:,1], marker='o', c=y, cmap=cm.get_cmap('plasma'))
    ax3.set_xlim(left=0.0, right=1.0)
    ax3.set_ylim(bottom=0.0, top=1.0)

    return fig

def train_plots(loss_dict, reg_coeff, epochs, start=0):
    fig = plt.figure(figsize=(27, 9))
    # plt.title(label=f'reg_coeff: {reg_coeff}', loc='center')

    axs1 = fig.add_subplot(1, 3, 1, title='Loss')
    axs2 = fig.add_subplot(1, 3, 2, title='Reconstruction Loss')
    axs3 = fig.add_subplot(1, 3, 3, title='Regularizer Loss')

    epoch_ax = [i for i in range(start, epochs)]

    sns.set_theme(style="whitegrid")

    sns.lineplot(x=epoch_ax, y=loss_dict['loss'][start: ], ax=axs1)
    sns.lineplot(x=epoch_ax, y=loss_dict['rec'][start: ], ax=axs2)
    sns.lineplot(x=epoch_ax, y=loss_dict['struc'][start: ], ax=axs3)

    return fig

def dataset_params(dataset): 
    fname, lyrs = None, None
    params = {
        'swiss': (
            'swiss_roll_raw_2000',
            [3, 48, 80, 32, 2]
        ),
        'cylinder': (
            'open_cylinder',
            [3, 9, 15, 6, 2]
        ),
        'scurve': (
            'S_Curve_raw_2000',
            [3, 36, 60, 24, 2]
        ),
        'helix': (
            'helix_2000',
            [3, 18, 30, 12, 2]
        ),
        'closed_curve': (
            '3D_Closed_Curve',
            [3, 48, 80, 32, 2]
        ),
        'uniform': (
            'Uniform_Random',
            [9, 45, 75, 30, 2]
        ),
        'smiley': (
            'smiley_5000',
            [9, 45, 75, 30, 2]
        ),
        'cassini': (
            'cassini_5000',
            [9, 45, 75, 30, 2]
        )
    }
    
    try:
        fname, lyrs = params[dataset]
    except KeyError as err:
        print(f'Invalid dataset key provided {dataset}')
        
    return fname, lyrs

def float_reps(float_val):
    if float_val == 1e-1:
        return "1e-1"
    elif float_val == 1e-2:
        return "1e-2"
    elif float_val == 1e-3:
        return "1e-3"
    elif float_val == 1e-4:
        return "1e-4"
    elif float_val == 1:
        return "1e0"    
    return str(float_val)

def slice_data(X, y, data):
    upp = {
        'swiss': 200,
        'cylinder': 1850,
        'scurve': 975
    }
    
    upper = upp[data]
    
    c = y[:].tolist()
    c_dexes = [[i, c[i][0]] for i in range(len(c))]
    c_dexes_sorted = sorted(c_dexes, key=lambda x: x[1], reverse=False)
    idxs = [i[0] for i in c_dexes_sorted]

    idxs_train = idxs[:X.shape[0] - upper] + idxs[X.shape[0] - upper + 50:]
    idxs_val = idxs[X.shape[0] - upper: X.shape[0] - upper+50]
    
    return idxs_train, idxs_val