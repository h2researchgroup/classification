<div id="top"></div>

<br />
<div align="center">

  <h2 align="center">Towards Computational Literature Reviews</h2>

  <p align="center">
     A collaborative data science project headed by the Massive Data Institute's Jaren Haber and UC Berkeley professor Heather Haveman.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#important-files">Guide to Codebase</a></li>
    <li><a href="#data-and-data-processing">Data And Data Processing</a></li>
    <li><a href="#model-training">Model Training</a></li>
    <li><a href="#analysis">Analysis</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Literature reviews are a vital part of research in many academic fields, and serve to help determine what we know and don’t know.  For interdisciplinary studies, literature reviews are especially challenging because the numerous publications and publication outlets are growing exponentially—critically, growing to a volume that a single person cannot easily manage. Our solution is to explore the capabilities of modern machine learning techniques to review the literature automatically at a large scale.


This code uses raw JSTOR OCR raw text, and creates predictive models based on hand-coded training data (per perspective) built on pretrained transformer models. We will then use these models to predict labels for papers at a large scale, and visualize overall trends across different perspectives over time.
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Guide to Codebase -->
## Guide to Codebase

#### Transformer-Based Approaches
* Scripts to train a Longformer Model on Azure:
    * <a href="modeling/Azure Files/Longformer-CV.py">modeling/Azure Files/Longformer-CV.py</a>
    * <a href="modeling/Azure Files/run.ipynb">modeling/Azure Files/run.ipynb</a>
* Notebooks to interactively train a BERT Model:
    * <a href="modeling/BERT-cross-validate.ipynb">modeling/BERT-cross-validate.ipynb</a>
* Notebook to label all 65k unlabeled articles using a Longformer model:
    * <a href="modeling/Longformer-Labeling.ipynb">modeling/Longformer-Labeling.ipynb</a>
* Text preprocessing code (updated to retain stopwords for transformers):
    * <a href="preprocess/preprocess_article_text_full.py">preprocess/preprocess_article_text_full.py</a>
    * <a href="preprocess/textprocess.ipynb">preprocess/textprocess.ipynb</a>

#### Machine Learning with `scikit-learn`, `keras`, and `gensim`
* Notebooks to build, evaluate, and optimize models with `scikit-learn`:
    * <a href="modeling/classifier_gridsearch.ipynb">modeling/classifier_gridsearch.ipynb</a>
    * <a href="modeling/evaluate_basic_classifiers_balanced.ipynb">modeling/evaluate_basic_classifiers_balanced.ipynb</a>
* Notebook to build and evaluate CNN and MLP models with `keras`:
    * <a href="modeling/mlp_train.py">modeling/mlp_train.py</a>
* Notebook to build and evaluate models with `scikit-learn` and word embeddings:
    * <a href="modeling/word_embedding_classification_cnn.ipynb">modeling/word_embedding_classification_cnn.ipynb</a>
* Notebook to build and evaluate CNN and MLP models with `keras` and word embeddings:
    * <a href="modeling/word_embedding_classification_mean.ipynb">modeling/word_embedding_classification_mean.ipynb</a>

#### Utilities
* Notebook to select and compile sample of articles across classes using model predictions:
    * <a href="modeling/sample_articles_with_models.ipynb">modeling/sample_articles_with_models.ipynb</a>
* Scripts with functions to assist in reading text files and text preprocessing:
    * <a href="preprocess/clean_text.py">preprocess/clean_text.py</a>
    * <a href="preprocess/text_to_file.py">preprocess/text_to_file.py</a>
* Notebook to load and merge datasets to assemble datasets for each perspective:
    * <a href="preprocess/assemble_coded_articles.ipynb">preprocess/assemble_coded_articles.ipynb</a>
* Notebook creating a csv of our file names:
    * <a href="modeling/grab filenames.ipynb">modeling/grab filenames.ipynb</a>
* CSV Logs of model hyperparameter and results for each perspective using the average word embedding as feature vector:
    * <a href="modeling/logs/ave_embed_cultural_grid_092621.csv">modeling/logs/ave_embed_cultural_grid_092621.csv</a>
    * <a href="modeling/logs/ave_embed_demographic_grid_092621.csv">modeling/logs/ave_embed_demographic_grid_092621.csv</a>
    * <a href="modeling/logs/ave_embed_orgs_grid_092621.csv">modeling/logs/ave_embed_orgs_grid_092621.csv</a>
    * <a href="modeling/logs/ave_embed_relational_grid_092621.csv">modeling/logs/ave_embed_relational_grid_092621.csv</a>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- DataProcessing -->
## Data And Data Processing

The data is obtained from approximately 70,000 JSTOR academic articles focusing on (1) sociology and (2) management and organizational behavior. Articles that are not full, in English-language, or book reviews are excluded. Our models are then trained and evaluated on hundreds of hand-labeled articles for each sociological perspective.

We preprocessed the text minimally. Because BERT utilizes structures of the sentence, stop words and grammatical structures become important as BERT assigns meaning to words based on their surrounding words in a process called self-attention. Thus, as part of the preprocessing step, we removed HTML and LaTeX tags in the JSTOR articles.

The script that processing the JSTOR files and saves the preprocessed files is in [preprocess/preprocess_article_text_full.py](preprocess/preprocess_article_text_full.py).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Model Training -->
## Model Training
Our current training script uses the Longformer as implemented by the Hugging Face library. For a given sociological perspective, our model conducts cross-validation training and evaluation on the perspective's dataset of labeled papers. An interactive example of this code for BERT is found in [modeling/BERT-cross-validate.ipynb](modeling/BERT-cross-validate.ipynb).

To train our models, we took advantage of our student Microsoft Azure computation credits. We used the platform’s machine learning servers to search the space of hyperparameters for our model to maximize cross-validation accuracy, utilizing their powerful cloud GPUs and well-used hyperparameter search framework. The files used to run these experiments are found in [modeling/Azure Files/](modeling/Azure%20Files/).

We have about 700 training data for each perspective, and have about twice as much negative data as positive data. To address the issue of class imbalance in our data, we used oversampling. To oversample, we bootstrap from the minority class (label 1) so that the ratio of majority and minority class is 1:1. We then perform the same procedure on the test data.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Analysis -->
## Analysis

We first ran the models over ~65K unlabeled JSTOR articles from 1970 to 2016 to obtain the predicted probability that each article is of the 4 perspectives. Then to start our analysis, we divided the articles into 2 primary subjects: Sociology and Management & Organizational Behavior, each having 3 perspectives (demographic, cultural, and relational). Sociology articles are filtered to be organizational sociology, where the predicted organizational score is greater than a threshold of 0.7. 

To analyze trends in various perspectives, we calculated the proportion of articles belonging to a certain perspective and primary subject and obtained the line graph above. The year 1970 and 2016 are outliers because both only contain less than 15 articles, resulting in the sharp fluctuations in some perspectives in 1970 and 2016. 
![Sociological Trends](https://github.com/h2researchgroup/classification/blob/figures/SOCTrends_Longformer_Dec2021.png?raw=true)

![Management Trends](https://github.com/h2researchgroup/classification/blob/figures/MGTTrends_Longformer_Dec2021.png?raw=true)

Overall, we see that demographic perspective is the most common in sociology articles, while relational perspective is the most common in management & OB articles. Demographic management articles seems to be in gradual decline starting in 2010, and cultural management articles seems to be in gradual growth starting in 1970. All other categories seem to have fluctuations over the years but exhibit no general growth nor decline from 1970 to 2016.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [UC Berkeley Data Science Discovery Program](http://dsdiscovery.org/)
* [Hugging Face Transformers Library](https://huggingface.co)
* [`scikit-learn` Machine Learning Library](https://scikit-learn.org/stable/)

<p align="right">(<a href="#top">back to top</a>)</p>




