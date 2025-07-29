import os
import openai
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, DATA_UNDEFINED_NAME

# Set your OpenAI API key as environment variable before running:
# export OPENAI_API_KEY="your-key"

openai.api_key = os.environ.get("OPENAI_API_KEY")

class GPTFinancialClassifier(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.from_name, self.to_name, self.value = get_single_tag_keys(
            self.parsed_label_config, 'Text'
        )
        self.labels = list(self.parsed_label_config[self.from_name]['labels'])

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini", # Or gpt-3.5-turbo if cheaper
                messages=[
                    {"role": "system", "content": "Classify emails as Financial or Not Financial."},
                    {"role": "user", "content": text}
                ]
            )
            label = "Financial" if "financial" in response['choices'][0]['message']['content'].lower() else "Not Financial"
            predictions.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {'choices': [label]}
                }]
            })
        return predictions
