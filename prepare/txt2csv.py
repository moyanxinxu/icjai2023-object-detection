import os

import pandas as pd

train_txt_folder = "../datasets/train/annotations"
test_img_folder = "../datasets/test/images"

txt2csv = '../info/data.csv'
label2id = '../info/label2id.txt'

columns = ["file_name", "category", "xmin", "ymin", "xmax", "ymax"]  # dataframe的索引列表
conver_dict = {"xmin": int, "ymin": int, "xmax": int, "ymax": int}  # 数据类型转换映射表


def read_from_txt(save_label2id=False):
    dataset = []
    for txt in os.listdir(train_txt_folder):
        txt_path = os.path.join(train_txt_folder, txt)
        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                token = line.strip().split(" ")
                token[0] = token[0].split(".")[0]
                dataset.append(token)

    dataset = pd.DataFrame(dataset, columns=columns).astype(conver_dict)
    dataset.file_name += ".jpg"
    dataset.set_index('file_name')
    dataset.category, unique_labels = pd.factorize(dataset.category)

    if save_label2id:
        with open(label2id, mode='w') as label_2id:
            for idx, row in enumerate(unique_labels):
                label_2id.write(f'{str(row)} {idx}\n')

    dataset.to_csv(txt2csv, index=False)


if __name__ == '__main__':
    read_from_txt(save_label2id=True)
