import os
from datetime import datetime


# create_directory function creates a directory if it does not exist.

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory to'{path}' was created successfully.")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")


# save_log_data function logs the training details to a file.

def save_log_data(save_path, model_name, batch_size, num_epochs, learning_rate, L2_regularisation):
    with open(save_path + model_name + '_log.txt', 'a') as file:
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the log entry
        log_entry = (f"\nTimestamp: {timestamp}\n"
                     f"Model Name: {model_name}\n"
                     f"Batch Size: {batch_size}\n"
                     f"Number of Epochs: {num_epochs}\n"
                     f"Learning Rate: {learning_rate}\n"
                     f"L2 Regularisation: {L2_regularisation}\n"
                     f"{'-'*40}\n")
        
        # Write the log entry to the file
        file.write(log_entry)


# get_input function gets user input and returns the default value if the input is invalid.

def get_input(prompt, default_value, value_type):
    user_input = input(prompt).strip()
    if not user_input:
        print(f"Using default value: {default_value}")
        return default_value
    if user_input.lower() == 'no':
        print(f"Using default value: {default_value}")
        return default_value
    try:
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default_value}")
        return default_value