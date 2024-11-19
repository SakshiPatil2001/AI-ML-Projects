# Next Word Prediction Using LSTM

This project demonstrates a next-word prediction model using an LSTM (Long Short-Term Memory) neural network. The model is trained on the text of *Alice's Adventures in Wonderland* to predict the next word in a sequence and generate sentences based on a given word or phrase.

## Features

- Preprocesses text for tokenization and sequence generation.  
- Builds and trains an LSTM-based neural network.  
- Predicts and generates text based on user input.  
- Supports dynamic sentence generation from any starting word or phrase.  

## Dataset

The dataset used is the text from *Alice's Adventures in Wonderland* by Lewis Carroll. Ensure the text file is available at the specified path in the script.

## Requirements

The project requires the following Python libraries: `numpy`, `tensorflow`, and `keras`. Install the required libraries using:  
`pip install numpy tensorflow`


## Usage

1. Clone the repository using `git clone <repository-link>` and navigate to the project folder using `cd Next Word Prediction`.  
2. Ensure the file `Alice's Adventures in Wonderland.txt` is available in the same directory.  
3. Execute the script `app.py` to train the model and test text generation using `python app.py`.  
4. Modify the `input_word` variable in the script to test text prediction. For example, set `input_word = "wonderland"` to see the output.

## Run the Code
python app.py

## Example Output

Input: `wonderland`  
Generated Sentence: `wonderland was full of strange and wonderful things as alice wandered through`

## Model Details

The model consists of an Embedding layer to convert words into dense vector representations, an LSTM layer to capture temporal dependencies in the text data, and a Dense layer to output a probability distribution over the vocabulary.

