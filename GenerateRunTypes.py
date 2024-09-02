import os
from openai import OpenAI

class GenerateRunTypes:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def generate(self, n_clusters, table):
        prompt = f"Below is a table of averages of the features of running statistics that have been clustered in {n_clusters} different categories of run types. Features used are: Distance, Calories, Time, Avg HR, Max HR, Aerobic TE, Avg Run Cadence, Max Run Cadence, Avg Pace, Best Pace, Total Ascent, Total Descent, Avg Stride Length, Min Temp, Number of Laps, Max Temp, Moving Time, Elapsed Time,  Min Elevation, Max Elevation. Please provide a unique and intuitive label for each of the run clusters based on the average statistics.\n###\nTABLE\n{table}"
        print(prompt)
        run_labels = self.__generate(prompt)

        prompt = f"Turn this list of run labels into a comma separated list.###\n{run_labels}"
        list_of_run_labels = self.__generate(prompt, 'gpt-3.5-turbo')

        return list_of_run_labels.split(",")

    def __generate(self, prompt, model='gpt-4o'):
        model_response = self.client.chat.completions.create(
          model=model,
          messages=[
            {
              "role": "user",
              "content": prompt
            }
          ],
          temperature=0.0,
          max_tokens=1024,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )

        return model_response.choices[0].message.content.strip('"')
