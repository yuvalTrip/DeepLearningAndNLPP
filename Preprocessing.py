import pandas as pd

# Load the data
data = pd.read_csv("heart-disease-extra.csv")  # Assuming your data is in a CSV file

# Replace 'No' with 0 and 'Yes' with 1 for binary categorical variables
binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})

# Convert 'Female' to 0 and 'Male' to 1 for the 'Sex' column
data['Sex'] = data['Sex'].replace({'Female': 0, 'Male': 1})

# Encode 'Race' into numerical values
race_mapping = {'Black': 0, 'White': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
data['Race'] = data['Race'].map(race_mapping)

# Encode 'Diabetic' into numerical values
diabetic_mapping = {'No': 0, 'Yes': 1, 'No, borderline diabetes': 0.5}
data['Diabetic'] = data['Diabetic'].map(diabetic_mapping)

# Encode 'GenHealth' into numerical values
genhealth_mapping = {'Excellent': 4, 'Very good': 3, 'Good': 2, 'Fair': 1, 'Poor': 0}
data['GenHealth'] = data['GenHealth'].map(genhealth_mapping)

# Convert 'AgeCategory' into real integer ages
age_mapping = {
    '0-4': 2,
    '5-9': 7,
    '10-14': 12,
    '15-19': 17,
    '20-24': 22,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80 or older': 85  # Taking the upper bound for this category
}
data['Age'] = data['AgeCategory'].map(age_mapping)

# Drop the 'AgeCategory' column if not needed anymore
data.drop(columns=['AgeCategory'], inplace=True)

# Export the preprocessed data to a new CSV file
data.to_csv("preprocessed_data.csv", index=False)