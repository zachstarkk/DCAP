import numpy as np
import pandas as pd
import torch.utils.data


class FrappeDataset(torch.utils.data.Dataset):
    """
    Frappe Dataset

    Data preparation
        treat apps with a rating less than 3 as negative samples

    :param dataset_path: frappe dataset path

    Reference:
        https://?
    """

    def __init__(self, dataset_path):
        super().__init__()
        df = pd.read_csv(dataset_path+'/frappe.csv', sep="\t")
        print("In the frappe data set we have {} entries by {} users for {} apps".format(len(df), len(df.user.unique()), len(df.item.unique())))
        meta_app = pd.read_csv(dataset_path+'/meta.csv', sep="\t")
        df = df.merge(meta_app, on='item')
        df = df[df["rating"]!='unknown']

        self.targets = df["rating"].to_numpy().astype(np.float32)
        self.FEATS = ["user", "item", "daytime", "weekday", "isweekend", "homework", "cost", "weather", "country", "city"]
        df = df[self.FEATS]
        self.items = self.__preprocess_items(df)
        self.items = self.items.to_numpy().astype(np.int)
        self.targets = self.__preprocess_target(self.targets).astype(np.float32)
        
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
        
    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    def __preprocess_items(self, df):
        for feature in ["daytime", "weekday", "isweekend", "homework", "cost", "weather", "country"]:
            df[feature] = pd.factorize(df[feature])[0]
        return df
