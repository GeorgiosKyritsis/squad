### Install poetry

Make sure that you have python3.8 installed.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install project

Inside the project's top level directory execute the following command:

```bash
poetry install
```

### Enable virtualenv

```bash
poetry shell
```

### Download dataset

```bash
mkdir -p data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -P data/
```

### Prepare the data

```bash
fire prepare_data --original_dataset_path data/dev-v2.0.json --output_directory data/with_impossible --include_impossible True
fire prepare_data --original_dataset_path data/dev-v2.0.json --output_directory data/without_impossible --include_impossible False
```

### Train the model

```bash
fire train data/with_impossible/train.json data/with_impossible/dev.json models/with_impossible/model.ckpt --batch_size 2 --learning_rate 0.0001 --accumulate_grad_batches 32
fire train data/without_impossible/train.json data/without_impossible/dev.json models/without_impossible/model.ckpt --batch_size 2 --learning_rate 0.0001 --accumulate_grad_batches 32
```

### Evaluate the model

```bash
fire evaluate data/with_impossible/test.json models/with_impossible/model.ckpt --batch_size 2
fire evaluate data/without_impossible/test.json models/without_impossible/model.ckpt --batch_size 2
```

### Tune the model

```bash
fire tune data/with_impossible/train.json data/with_impossible/dev.json results/with_impossible/tuning_results.pkl
fire tune data/without_impossible/train.json data/without_impossible/dev.json results/without_impossible/tuning_results.pkl
```

### Run tests

```bash
pytest
```

## Web application

[Streamlit](https://streamlit.io) was used to build the frontend and [FastAPI](https://fastapi.tiangolo.com) for the backend (uvicorn was used to run  it). [Pydantic](https://pydantic-docs.helpmanual.io) was used for data validation. The code can be found in the webapp folder.

To launch the FastAPI

```bash
nohup python3 webapp/back.py > back.out &
```

To launch the Streamlit application

```bash
nohup streamlit run webapp/front.py > front.out &
```

The application is deployed on an EC2 r5.large instance on AWS.

It can be found here: [http://18.184.83.51:8501](http://18.184.83.51:8501). Feel free to use it!!!

The Swagger UI can be found here: [http://18.184.83.51:1111/docs](http://18.184.83.51:1111/docs)

## Data Preparation

I shuffled the original dataset and I split into 3 sets, namely: training, development and test sets. From the 35 documents that the original dataset contains, I put 25 documents to the training set, 5 documents to the development set and 5 documents to the test set. I split based on the documents because every document contains related paragraphs, that could have a negative impact on the generalization capabilities of the model if the split was done on the paragraph level. The model should be evaluated on new paragraphs.
During the training phase the training and development sets were used, the development set was used for early stopping. Also the development set is used during the hyperparameter tuning where I grid search on the learning rate and the batch size. Finally, the test set is used to assess how well the model can generalize on new unseen data. 

To cover both SQuAD dataset versions I used the SQuAD2.0 dataset which is a superset of SQuAD1.0 dataset and trained 2 models.

My first model which is called “with_impossible” includes unanswerable questions. To handle this case I trained the model to generate the token “null” when there is no answer.

My second model which is called “without_impossible” does not include the unanswerable questions, so I removed them from the datasets before feeding them to the network.

## Architecture

In our task a pre-trained encoder-decoder T5 generative model was used. More concretely I used the T5 Version 1.1 which was trained only on the C4 corpus excluding any supervised training, as opposed to version 1.0 which was also trained on many tasks including SQuAD.
During the T5 v1.1 training all tasks were formulated as text-to-text tasks. One goal was to generate sequences of tokens that were masked in the original text.

In order to use this pre-trained architecture, we need to formulate the SQuAD dataset as follows before feeding it to the network: we append the question to the context and we mask the answer that follows the question, which we try to generate.

## Training

I used the Subclassing API from Tensorflow v2 to build my architecture.
With this API I subclassed the tf.keras.Model and I created the layers that I needed in the constructor and I used them in the call() method.
Also I trained my model with Gradient Accumulation by overriding the train_step() method with a custom training loop.
I used this method to reduce the memory usage while training the model. With this technique the model parameters are not updated after every batch but they are accumulated and aggregated after some predefined steps.
The training was done on an NVIDIA GeForce RTX 2080Ti GPU. 

## Results

#### with_impossible Model (unanswerable questions)

|    | Development Set   | Test Set          |
|----|-------------------|-------------------|
| F1 | 66.9737773119156  | 66.11491962772038 |
| EM | 64.11645226811103 | 63.2641291810842  |

#### without_impossible Model (answerable questions only)

|    | Development Set   | Test Set          |
|----|-------------------|-------------------|
| F1 | 82.88712890640477 | 73.16181883237513 |
| EM | 74.14012738853503 | 63.00863131935882 |

### Models Weights

The models' weights can be found here: [https://www.dropbox.com/sh/stgi5tade1bto8l/AABxCHm4Itt7WciP5cV7vf2Oa?dl=0](https://www.dropbox.com/sh/stgi5tade1bto8l/AABxCHm4Itt7WciP5cV7vf2Oa?dl=0)
Put them in the models folders and then you can evaluate the model.

### Datasets

The original dataset and the splits for the 2 models can be found here: [https://www.dropbox.com/sh/stgi5tade1bto8l/AABxCHm4Itt7WciP5cV7vf2Oa?dl=0](https://www.dropbox.com/sh/stgi5tade1bto8l/AABxCHm4Itt7WciP5cV7vf2Oa?dl=0)
You should put them in the data folder and you can train a model or tune it.
