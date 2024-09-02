# Running Coach

## Motivation

I would like to use the data from my past runs to craft training plans tailored to me to accomplish my running goals.


## Notes and Ideas

- Download all runs from Garmin Connect
- Get the run types from TrainingPeaks
- Categorize the run types using OpenAI in 5 groups:
  - Base Building and Aerobic Development
  - Speed and Anaerobic Development
  - Lactate Threshold and Tempo Training
  - Race Simulation and Pacing
  - Variety and Mixed Intensity
- Build a model based on the runs that are already categorized
- Do the same for the Feeling and TSS scores from TrainingPeaks
- Then use that model to label the rest of the runs (most should be Base Building)
- Then use the dataset to train a model to predict
  - Average Pace
  - Recommended Pace
  - Recommended Run Type
  - Expected Aerobic Training Effect (Aerobic TE)