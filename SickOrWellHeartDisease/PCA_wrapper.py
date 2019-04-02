from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
import numpy as np


# Standard PCA

def number_of_components_pca(dataset_to_reduce, number_of_components):
    pca = PCA(n_components=number_of_components)
    return pca.fit_transform(dataset_to_reduce), pca


def variance_pca(dataset_to_reduce, variance_ratio):
    pca = PCA(n_components=variance_ratio)
    return pca.fit_transform(dataset_to_reduce), pca


def explained_variance_vs_dimensions_pca(dataset_to_reduce):
    pca = PCA()
    pca.fit(dataset_to_reduce)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return cumsum


# Incremental PCA

def number_of_components_incremental_pca(dataset_to_reduce, number_of_components, n_batches=100):
    inc_pca = IncrementalPCA(n_components=number_of_components)
    for X_batch in np.array_split(dataset_to_reduce, n_batches):
        inc_pca.partial_fit(X_batch)

    return inc_pca.transform(dataset_to_reduce)


def variance_incremental_pca(dataset_to_reduce, variance_ratio, n_batches=100):
    inc_pca = IncrementalPCA(n_components=variance_ratio)

    for X_batch in np.array_split(dataset_to_reduce, n_batches):
        inc_pca.partial_fit(X_batch)

    return inc_pca.transform(dataset_to_reduce)


def explained_variance_vs_dimensions_incremental_pca(dataset_to_reduce, n_batches=100):
    inc_pca = IncrementalPCA()

    for X_batch in np.array_split(dataset_to_reduce, n_batches):
        inc_pca.partial_fit(X_batch)

    cumsum = np.cumsum(inc_pca.explained_variance_ratio_)
    return cumsum


# Randomized PCA

def number_of_components_rnd_pca(dataset_to_reduce, number_of_components):
    rnd_pca = PCA(n_components=number_of_components, svd_solver="randomized")
    return rnd_pca.fit_transform(dataset_to_reduce), rnd_pca


def variance_rnd_pca(dataset_to_reduce, variance_ratio):
    rnd_pca = PCA(n_components=variance_ratio, svd_solver='randomized')
    return rnd_pca.fit_transform(dataset_to_reduce), rnd_pca


def explained_variance_vs_dimensions_randomized_pca(dataset_to_reduce):
    rnd_pca = PCA(svd_solver="randomized")
    rnd_pca.fit(dataset_to_reduce)
    cumsum = np.cumsum(rnd_pca.explained_variance_ratio_)
    return cumsum


# Kernel PCA

def number_of_components_kernel_pca(dataset_to_reduce, number_of_components, kernel_='rbf', gamma_=0.0433,
                                    fit_inverse_transform_=True):
    kernel_pca = KernelPCA(n_components=number_of_components, kernel=kernel_, gamma=gamma_,
                            fit_inverse_transform=fit_inverse_transform_)
    return kernel_pca.fit_transform(dataset_to_reduce), kernel_pca


def variance_kernel_pca(dataset_to_reduce, variance_ratio, kernel_='rbf', gamma_=0.0433,
                        fit_inverse_transform_=True):
    kernel_pca = KernelPCA(n_components=variance_ratio, kernel=kernel_, gamma=gamma_,
                        fit_inverse_transform=fit_inverse_transform_)
    return kernel_pca.fit_transform(dataset_to_reduce), kernel_pca


# Functions for all pcas

def return_selected_component(pca, n_component):  # Except kernel :p
    return pca.components_.T[:, n_component]


def calculate_compression(original_dataset, reduced_dataset):
    return len(reduced_dataset)/len(original_dataset)


def print_explained_variance_ratio(pca):  # Except kernel :p
    print(str(pca.explained_variance_ratio_))


def recover_dataset(dataset_to_recover, pca):
    return pca.inverse_transform(dataset_to_recover)


def find_number_of_dimensions_explaining_variance(cumsum, desired_variance_explained):
    dimensions_explaining_variance = np.argmax(cumsum >= desired_variance_explained) + 1
    return dimensions_explaining_variance