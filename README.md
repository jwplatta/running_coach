# Running Coach AI

## Goals

- I would like to know exactly what runs and workouts I need to do in order to run a sub 3 hour marathon.
- So I guess I would like to know my progress towards getting to the right fitness level to accomplish this.
- So maybe recommend a run that targets ATE between 3.0 and 4.0 for key workouts.
- So maybe an MVP is a dropdown that lets me select Run Type and predicts the ATE and Feeling scores based on the Run Type.
- Previous runs should affect the quality of your next run and therefore the recommendation.
- Would be great to include previous runs as features and your previous night's sleep score and even nutrition.

## Notes

## Building the Dataset

- Download all runs from Garmin Connect
- Get the run types from TrainingPeaks
- Categorize the run types using OpenAI in 5 groups:
  - Base Building and Aerobic Development
  - Speed and Anaerobic Development
  - Lactate Threshold and Tempo Training
  - Race Simulation and Pacing
  - Variety and Mixed Intensity
- Build a model based on the runs that are already categorized
  - Use KNN probably
- Do the same for the Feeling and TSS scores from TrainingPeaks
  - Use KNN probably
- Then use that model to label the rest of the runs (most should be Base Building)
- Then use the dataset to train a model to predict
  - Average Pace
  - Recommended Pace
  - Recommended Run Type
  - Expected Aerobic Training Effect (Aerobic TE)

## Training Models

## Design

- Can use gradio for the interface
- dockerize the app and deploy to
  - heroku no longer has a free tier
  - hugging face spaces
  - streamlit