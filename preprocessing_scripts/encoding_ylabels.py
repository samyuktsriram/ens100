import pandas as pd

# Define the encodings dictionary
encodings = {
    'GOOD': 0,
    'Boucle plate': 1,
    'Lift-off blanc': 2,
    'Lift-off noir': 3,
    'Missing': 4,
    'Short circuit MOS': 5
    # Add more label encodings as needed
}

# Path to the CSV file
csv_file_path = '/mnt/disks/location/Y_train.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Specify the column to encode
column_to_encode = 'Label'

# Encode the labels using the encodings dictionary
df[column_to_encode] = df[column_to_encode].map(encodings)

# Save the encoded DataFrame back to a CSV file
df.to_csv('/mnt/disks/location/Y_train.csv', index=False)