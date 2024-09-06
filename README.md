# Classification of App Reviews using PLMs

In this project, classification of mobile app reviews was done into four categories: bug report, feature request, user experience and rating. The dataset used for this project is the [App Reviews dataset by Pavel GH](https://huggingface.co/datasets/PavelGh/app_reviews) available on Hugging Face.

Three Pre-Trained Language Models(PLMs) were fine-tuned on this dataset. For fine-tuning, the Parameter Efficient Fine-Tuning(PEFT) technique was used due to limited computational resources and for faster training purposes. The 3 PLMs used as base models for this classification task were DistilBERT, ALBERT, and BERT4RE.

The models are saved in root directory of this project.

# GUI for testing the models
Before testing the GUI, please install the required packages in your local device or virtual env. You can use the following command for the same:

`pip3 install -r requirements.txt` 

After installing the requirements you can use the following command to run the Flask application and test the models by opening the Flask app in your browser.
