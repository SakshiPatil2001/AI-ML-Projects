# Import necessary libraries
import pandas as pd
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the data
file_path = r"C:\Users\saksh\OneDrive\Desktop\Google Rewiew Card\Untitled form.csv"
data = pd.read_csv(file_path)

# Clean column names by stripping any leading or trailing spaces
# This ensures that column names are properly formatted for further operations
data.columns = data.columns.str.strip()

# Initialize T5 model and tokenizer
# Here, the T5 model is loaded using the 't5-small' pre-trained version for text generation
model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate a review using T5 model with random selection of questions
def generate_t5_review(row):
    # Define the questions and corresponding templates
    # Each category from the dataset has predefined templates that will be filled with the user's input
    questions_templates = {
        'project_type': [
            f"I had a {row['What type of elevator project did Horizon Elevators complete for you?']} with Horizon Elevators.",
            f"My experience with Horizon Elevators was for a {row['What type of elevator project did Horizon Elevators complete for you?']} project.",
            f"Had a {row['What type of elevator project did Horizon Elevators complete for you?']} completed by Horizon Elevators."
        ],
        'responsiveness': [
            f"They were {row['How responsive has Horizon Elevators been to your inquiries or concerns?']} to my inquiries and concerns.",
            f"They showed {row['How responsive has Horizon Elevators been to your inquiries or concerns?']} to my concerns.",
            f"It was a {row['How responsive has Horizon Elevators been to your inquiries or concerns?']} experience regarding my inquiries and concerns."
        ],
        'communication': [
            f"Throughout the process, I was {row['How well did Horizon Elevators keep you informed throughout the process (sales, installation, etc.)?']}.",
            f"From start to finish, their communication was {row['How well did Horizon Elevators keep you informed throughout the process (sales, installation, etc.)?']}.",
            f"Throughout, they kept me {row['How well did Horizon Elevators keep you informed throughout the process (sales, installation, etc.)?']}."
        ],
        'pricing': [
            f"Their pricing was {row['Compared to other elevator companies; did Horizon Elevators offer competitive pricing?']}.",
            f"In terms of pricing, they were {row['Compared to other elevator companies; did Horizon Elevators offer competitive pricing?']}.",
            f"Pricing-wise, they were {row['Compared to other elevator companies; did Horizon Elevators offer competitive pricing?']}."
        ],
        'amc_satisfaction': [
            f"I am {row['How satisfied are you with the Annual Maintenance Contract (AMC) service provided by Horizon Elevators?']} with the Annual Maintenance Contract (AMC) service.",
            f"The AMC service left me feeling {row['How satisfied are you with the Annual Maintenance Contract (AMC) service provided by Horizon Elevators?']}.",
            f"The AMC service has been {row['How satisfied are you with the Annual Maintenance Contract (AMC) service provided by Horizon Elevators?']}."
        ],
        'installation_professionalism': [
            f"The installation was {row['How professional and efficient was the installation process conducted by Horizon Elevators?']}.",
            f"The installation process was {row['How professional and efficient was the installation process conducted by Horizon Elevators?']}.",
            f"The professionalism during installation was {row['How professional and efficient was the installation process conducted by Horizon Elevators?']}."
        ],
        'overall_satisfaction': [
            f"I am {row['How satisfied are you with the overall quality, reliability, and safety of Horizon Elevators\' installations (if applicable) or maintenance services?']} with the quality, reliability, and safety of their installations.",
            f"I am {row['How satisfied are you with the overall quality, reliability, and safety of Horizon Elevators\' installations (if applicable) or maintenance services?']} with their service quality, reliability, and safety standards.",
            f"Overall, I'm {row['How satisfied are you with the overall quality, reliability, and safety of Horizon Elevators\' installations (if applicable) or maintenance services?']} with the quality, reliability, and safety of their work."
        ],
        'recommendation': [
            f"I would {row['Would you recommend Horizon Elevators to others looking for elevator installation or maintenance services?']} Horizon Elevators to others.",
            f"I would {row['Would you recommend Horizon Elevators to others looking for elevator installation or maintenance services?']} them to anyone in need of elevator services."
        ]
    }

    # Randomly select a subset of questions to include in the review
    # This creates more varied and personalized reviews
    selected_questions = random.sample(list(questions_templates.keys()), random.randint(3, len(questions_templates)))

    # Combine the selected templates into a full review by choosing a random template for each selected question
    review_template = " ".join(random.choice(questions_templates[q]) for q in selected_questions)

    return review_template

# Apply the function to the dataset to generate the reviews
# The 'apply' method runs the 'generate_t5_review' function on each row of the dataset
data['Generated Review'] = data.apply(generate_t5_review, axis=1)

# Save the generated reviews to a new CSV file
# This saves the generated reviews in a new file on your computer
output_file_path = r"C:\Users\saksh\OneDrive\Desktop\Google Rewiew Card\output.csv"
data.to_csv(output_file_path, index=False)

# Print a success message to confirm that the reviews have been saved
print(f"Generated Google reviews saved to {output_file_path}")


