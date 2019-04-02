import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report

import PCA_wrapper as pw
import heart_disease_uci_datarow as hdud
import csv_reader as cr
import clf_wrapper as cw
import conf_mat_wrapper as cmat_w


# TODO:
# Scale features

def load_dataset(filepath):
    return cr.load_csv(filepath=filepath)


def put_dataset_into_hdud(dataset_dataframe_datarows):
    dataset_datarows =[]

    for row in dataset_dataframe_datarows:
        dataset_datarows.append(hdud.HeartDiseaseUCIDatarow(row))

    return dataset_datarows


def reduce_dimensions_in_range(dataset_dataframe_datarows, end_number_of_dimensions,
                               start_number_of_dimensions=1):
    dataset_reduced = []
    PCAs = []

    for i in range(start_number_of_dimensions, end_number_of_dimensions + 1):
        dataset_reduced_row, pca = pw.number_of_components_pca(dataset_to_reduce=dataset_dataframe_datarows,
                                                               number_of_components=i)
        dataset_reduced.append(dataset_reduced_row)
        PCAs.append(pca)

    return dataset_reduced, PCAs


def plot_explained_variance_vs_dimensions(dataset_dataframe_datarows, cumsum):
    plt.title('Explained variance vs dimensions')
    plt.plot(cumsum)
    plt.show()


def retrieve_y_pred_y_test_for_proba_based_binary_clf(clf, X_test, y_test):
    y_pred_proba = clf.predict_proba(X_test)

    y_pred = list()
    for element in y_pred_proba:
        if element[0] > 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    y_test = [int(x) for x in y_test]

    return y_pred, y_test


def plot_conf_matrices(clf, cm, y_pred, y_test, title_annotation="full"):
    # Plot non-normalized confusion matrix
    cmat_w.plot_confusion_matrix(cm, y_test, y_pred, classes=class_names,
                                 title='Confusion matrix, without normalization - ' + title_annotation + ' dim set '
                                       + str(clf)[:10])
    plt.show()

    # Plot normalized confusion matrix
    cmat_w.plot_confusion_matrix(cm, y_test, y_pred, classes=class_names,
                                 normalize=True,
                                 title='Confusion matrix, with normalization - ' + title_annotation + ' dim set '
                                 + str(clf)[:10])
    plt.show()

if __name__ == "__main__":
    filepath = 'heart-disease-uci/heart.csv'
    dataset_dataframe = load_dataset(filepath)

    dataset_dataframe_datarows = dataset_dataframe._get_values

    target = list()
    dataset_dataframe_datarows_temp = list()
    for row in dataset_dataframe_datarows:
        target.append(row[-1])  # disease present or not
        dataset_dataframe_datarows_temp.append(row[:-1])

    dataset_dataframe_datarows = dataset_dataframe_datarows_temp
    del dataset_dataframe_datarows_temp

    # Useless - but I wanted to implement class :p
    # dataset_datarows = put_dataset_into_hdud(dataset_dataframe_datarows)

    dataset_reduced, PCAs = reduce_dimensions_in_range(dataset_dataframe_datarows, len(dataset_dataframe_datarows[0]))

    cumsum = pw.explained_variance_vs_dimensions_pca(dataset_dataframe_datarows)

    desired_variance_explained = 0.95
    dimensions_explaining_variance = pw.find_number_of_dimensions_explaining_variance(cumsum,
                                                                                      desired_variance_explained)

    print('How many dimensions explain ' + str(desired_variance_explained) + ' variance? '
          + str(dimensions_explaining_variance))

    X_train, X_test, y_train, y_test = cw.split_dataset(np.array(dataset_dataframe_datarows), target)
    split_dict_dataset_full = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    SVC_clf = cw.fit_SVC(np.array(split_dict_dataset_full["X_train"]),
                         split_dict_dataset_full["y_train"])
    SGD_clf = cw.fit_SGD(np.array(split_dict_dataset_full["X_train"]),
                         split_dict_dataset_full["y_train"])
    RandomForest_clf = cw.fit_RandomForrest(np.array(split_dict_dataset_full["X_train"]),
                         split_dict_dataset_full["y_train"])
    DecisionTree_clf = cw.fit_DecisionTree(np.array(split_dict_dataset_full["X_train"]),
                         split_dict_dataset_full["y_train"])

    dataset_reduced_desired_variance_explained = dataset_reduced[dimensions_explaining_variance - 1]
    X_train, X_test, y_train, y_test = cw.split_dataset(np.array(dataset_reduced_desired_variance_explained), target)
    split_dict_dataset_reduced = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    SVC_clf_reduced_dataset = cw.fit_SVC(np.array(split_dict_dataset_reduced["X_train"]),
                                         split_dict_dataset_reduced["y_train"])
    SGD_clf_reduced_dataset = cw.fit_SGD(np.array(split_dict_dataset_reduced["X_train"]),
                                         split_dict_dataset_full["y_train"])
    RandomForest_clf_reduced_dataset = cw.fit_RandomForrest(np.array(split_dict_dataset_reduced["X_train"]),
                                         split_dict_dataset_full["y_train"])
    DecisionTree_clf_reduced_dataset = cw.fit_DecisionTree(np.array(split_dict_dataset_reduced["X_train"]),
                                         split_dict_dataset_full["y_train"])


    print(SGD_clf.score(split_dict_dataset_full["X_test"], split_dict_dataset_full["y_test"]))
    print(SGD_clf_reduced_dataset.score(split_dict_dataset_reduced["X_test"], split_dict_dataset_reduced["y_test"]))

    print(SGD_clf.score(split_dict_dataset_full["X_train"], split_dict_dataset_full["y_train"]))
    print(SGD_clf_reduced_dataset.score(split_dict_dataset_reduced["X_train"], split_dict_dataset_reduced["y_train"]))

    print(SGD_clf.score(dataset_dataframe_datarows, target))
    print(SGD_clf_reduced_dataset.score(dataset_reduced_desired_variance_explained, target))

    class_names = [0, 1]
    precisions = list()
    recalls = list()
    accuracies = list()

    for clf in [SGD_clf, SVC_clf, RandomForest_clf, DecisionTree_clf]:
        y_pred, y_test = retrieve_y_pred_y_test_for_proba_based_binary_clf(clf,
                                                                       np.array(split_dict_dataset_full["X_test"]),
                                                                       np.array(split_dict_dataset_full["y_test"]))
        cm = cmat_w.compute_confusion_matrix(y_test, y_pred)
        plot_conf_matrices(clf, cm, y_pred, y_test)
        classification_report(y_test, y_pred)

        precision, recall = cmat_w.precision_recall_values(cm)
        clf_name = str(clf)[:15]
        precisions.append({"name": clf_name, "value": precision})
        recalls.append({"name": clf_name, "value": recall})
        accuracies.append({"name": clf_name, "value": cmat_w.accuracy_value(cm)})

    for clf_reduced in [SGD_clf_reduced_dataset, SVC_clf_reduced_dataset,
                        RandomForest_clf_reduced_dataset, DecisionTree_clf_reduced_dataset]:
        y_pred, y_test = retrieve_y_pred_y_test_for_proba_based_binary_clf(clf_reduced,
                                                                           np.array(split_dict_dataset_reduced["X_test"]),
                                                                           np.array(split_dict_dataset_reduced["y_test"]))
        cm = cmat_w.compute_confusion_matrix(y_test, y_pred)
        plot_conf_matrices(clf_reduced, cm, y_pred, y_test, "reduced")
        classification_report(y_test, y_pred)

        precision, recall = cmat_w.precision_recall_values(cm)
        clf_name = str(clf)[:15] + " reduced"
        precisions.append({"name": clf_name, "value": precision})
        recalls.append({"name": clf_name, "value": recall})
        accuracies.append({"name": clf_name, "value": cmat_w.accuracy_value(cm)})

    precisions_values = [element["value"] for element in precisions]
    recalls_values = [element["value"] for element in recalls]
    accuracies_values = [element["value"] for element in accuracies]

    for unsorted_list in [precisions, recalls, accuracies]:
        sorted_list = sorted(unsorted_list, key=lambda x: x["value"], reverse=True)
        print(str(sorted_list))

    print("Best accuracy: " + str(accuracies[accuracies_values.index(max(accuracies_values))]))
    print("Best precision: " + str(precisions[precisions_values.index(max(precisions_values))]))
    print("Best recall: " + str(recalls[recalls_values.index(max(recalls_values))]))
