# App Review Classification using Pre-Trained Language Models

This project classifies app reviews into four categories: bug report, feature request, user experience, and rating. It uses three pre-trained language models: **BERT4RE**, **ALBERT**, and **DistilBERT**. The models are fine-tuned using the **Parameter-Efficient Fine-Tuning (PEFT)** method and optimized via grid search for hyperparameters.

## Models & Performance
- **Models**: BERT4RE, ALBERT, and DistilBERT
- **Dataset**: [App Review Classification Dataset by Pavel Ghazaryan](https://huggingface.co/datasets/PavelGh/app_reviews)
- **Performance**: All models achieved an F1-score of at least 89% with optimal hyperparameters.

## Project Structure
- **Fine-Tuning Results**: Found in files with `PEFT` in the file name, located in the root directory.
- **Grid Search Results**: Found in files with `PEFT_Grid_Search` in the file name, located in the root directory.
- **Balanced Dataset Models**: Code for models trained on balanced datasets is in the `Balanced Dataset` folder.
- **Multi-Label Dataset Models**: Code for models trained on unbalanced multi-label datasets is in the `Multi-Label` folder.

## Classification Tool
A **Flask-based tool** allows users to:
- Input a single review for predictions from all three models.
- Upload multiple reviews in an Excel file, generating predictions in separate Excel files for each model.

## Flask App Screenshots

Here are some screenshots of the Flask app in action:

### Home Page
![gui_home](https://github.com/user-attachments/assets/ea129a06-e3dd-4c3e-b214-ce4f62e9d2c1)


### Single Review Prediction
![gui_single_review_results](https://github.com/user-attachments/assets/d115e60b-07cd-494b-86c6-9a32c05f9a47)
![gui_single_user_input](https://github.com/user-attachments/assets/eb278ff4-4fcb-4710-9d80-47727b1e65c3)


### Multiple Reviews Upload
![gui_excel_input](https://github.com/user-attachments/assets/a36ec910-43c9-4dd2-b5ac-1a3081c6cc6f)
![gui_excel_results](https://github.com/user-attachments/assets/dd4de260-9312-4d42-a9c1-f7140b6567cc)

## Running the Flask App

1. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:
    ```bash
    python main.py
    ```

## Requirements
All necessary dependencies are listed in the `requirements.txt` file.

## File Structure
1. Fine-tuning results for the unbalanced dataset: `PEFT_*` files in the root directory.
2. Grid search results for hyperparameters: `PEFT_Grid_Search_*` files in the root directory.
3. Balanced dataset models: `Balanced Dataset/` folder.
4. Unbalanced multi-label models: `Multi-Label/` folder.
5. Required packages: `requirements.txt`.
