import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import config as conf


def balance_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    positives_size = dataset["target"].value_counts()[1]

    positive_df = dataset[dataset["target"] == 1]
    negative_df = dataset[dataset["target"] == 0]

    negative_df = negative_df.sample(positives_size)

    balanced_df = pd.concat([positive_df, negative_df])

    return shuffle(balanced_df)


def get_train_data() -> tuple:
    train_df = pd.read_csv("datasets/train.csv")
    # train_df = balance_dataset(train_df)

    return dataset_to_numpy(train_df)


def get_validation_data() -> tuple:
    valid_df = pd.read_csv("datasets/valid.csv")

    return dataset_to_numpy(valid_df)


def get_test_data() -> tuple:
    test_df = pd.read_csv("datasets/test.csv")

    return dataset_to_numpy(test_df)


def dataset_to_numpy(dataset: pd.DataFrame) -> tuple:
    y = dataset["target"].to_numpy()
    x = dataset.drop(["target", "sample_type"], axis=1).to_numpy()

    # pca = PCA(conf.TARGET_SIZE)
    # x = pca.fit_transform(x)
    y = np.expand_dims(y, axis=1)

    return x, y


if __name__ == "__main__":
    x_train, y_train = get_train_data()
    x_valid, y_valid = get_validation_data()
    x_test, y_test = get_test_data()
    ...
