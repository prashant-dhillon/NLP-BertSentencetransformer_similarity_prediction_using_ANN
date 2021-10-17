from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

import os
import pickle
import scann
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter


class PipeLine():
    """
    Prediction pipeline to serve below objectives
    a. Vectorize the training set with your fine-tuned sBERT model
    b. Index all these vectors with your ANN library
    c. Build a barebone kNN classifier where new text input gets predicted the same label as that of the closest neighbor from the index
    d. Benchmark this pipeline with the test set
    e. Compare the results with the pretrained sBERT model
    """
    def __init__(self):
        """
        construct pipeline object, load models and embeddings. 
        """
        self.pretrained_model = self.load_model(
            "distilbert-base-nli-mean-tokens")
        self.finetuned_model = self.load_model(
            "./required/fine_tune_distilbert_nli_mean_token_")
        self.load_embeddings()

    def load_embeddings(self):
        """
        if embeddingd for ANN already exists load them else create embeddings from traindataset.
        """
        if not os.path.exists("./required/utils.pkl"):
            self.indexes, self.labels, self.target_names = self.build_index()
        else:
            with open("./required/utils.pkl", "rb") as f:
                utils = pickle.load(f)
            k = int(np.sqrt(utils["normalized_embeddings"].shape[0]))
            self.indexes = scann.scann_ops_pybind.builder(utils["normalized_embeddings"], 10, "dot_product").tree(
                num_leaves=k, num_leaves_to_search=int(k/20), training_sample_size=2500).score_brute_force(2).reorder(7).build()
            self.labels = utils["labels"]
            self.target_names = utils["target_names"]

    def load_model(self, model_name):
        model = SentenceTransformer(model_name)
        return model

    def build_index(self):
        """
        Using scann ANN algorithm build indexes for sentence embeddings to serch approximate nearest neighbours
        """
        newsgroups_train = fetch_20newsgroups(
            subset='train', remove=("headers", "footers", "quotes"))
        target_names = newsgroups_train.target_names
        news_text = newsgroups_train.data
        labels = newsgroups_train.target
        model = self.load_model(model_name="distilbert-base-nli-mean-tokens")
        train_sentence_embeddings = model.encode(news_text)

        normalized_embeddings = train_sentence_embeddings / \
            np.linalg.norm(train_sentence_embeddings, axis=1)[:, np.newaxis]
        # configure ScaNN as a tree - asymmetric hash hybrid with reordering
        # anisotropic quantization as described in the paper; see README
        k = int(np.sqrt(normalized_embeddings.shape[0]))
        # use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
        indexes = scann.scann_ops_pybind.builder(normalized_embeddings, 10, "dot_product").tree(
            num_leaves=k, num_leaves_to_search=int(k/20), training_sample_size=2500).score_brute_force(2).reorder(7).build()

        return indexes, labels, target_names

    def barebone_knn(self, df):
        """
        barebore knn to classify the class to the nearest neighbour from ANN index
        """
        counter = Counter(df.labels)
        res_class = self.target_names[counter.most_common()[0][0]]
        return res_class

    def predict(self, text, model=None):
        """
        predict the class based on input text
        """
        if not model:
            model = self.finetuned_model
        test_sentence_embedding = model.encode(text)
        # normalized_embedding_test = test_sentence_embedding / np.linalg.norm(test_sentence_embedding)[:, np.newaxis]

        neighbors, distances = self.indexes.search(
            test_sentence_embedding, final_num_neighbors=10)
        d_inf = {'neighbour': neighbors, 'distance': distances}
        df_inf = pd.DataFrame(data=d_inf)
        df_inf['labels'] = df_inf['neighbour'].apply(
            lambda row: self.labels[row])
        # df_inf['target_class'] = df_inf['labels'].apply(lambda row: target_names[row])

        result = self.barebone_knn(df_inf)
        # print(df_inf)
        return result

    def benchmark(self, model_name, test_set):
        """
        Based on model name and test set gives accuracy of the model
        """
        # Accucacy on the test set
        if model_name == "distilbert-base-nli-mean-tokens":
            model = self.pretrained_model
        else:
            model = self.finetuned_model

        true_label = list(test_set.target)
        count = 0
        for idx, text in enumerate(tqdm(test_set.data)):
            pred = self.predict(text, model)
            # print(self.target_names, type(self.target_names))
            predicted_label = self.target_names.index(pred)
            if predicted_label == true_label[idx]:
                count += 1
        accuracy = count / len(true_label)
        print("\nApproximate Nearest Neighbor precision on 20news group test with {} models: {:.2f}\n".format(model_name, accuracy * 100))
        return accuracy*100

    def benchmark_pipeline(self):
        """
        benchmarks both the model pretrained and finetuned model
        """
        newstestset = fetch_20newsgroups(
            subset='test', remove=('headers', 'footers', 'quotes'))

        model_paths = ['finetune-distilbert-base-nli-mean-tokens',
                       'distilbert-base-nli-mean-tokens']

        benchs = []
        for model_path in model_paths:
            # benchs[model_name] = benchmark(model_path, model_name, newstrainset, newstestset)
            benchs.append(self.benchmark(model_path, newstestset))
        return benchs
