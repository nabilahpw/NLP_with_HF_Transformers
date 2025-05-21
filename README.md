<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Nabilah Putri Wijaya 
## UNIVERSITAS SEMARANG

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
original_model = pipeline("sentiment-analysis")
data = "The rapid growth of electric vehicles is reshaping the automotive industry, but concerns about battery production and environmental impact still remain."
original_model(data)

```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.9938055276870728}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "The recent advancements in renewable energy technologies are shaping the future of sustainable development.",
    candidate_labels=["environment", "energy", "climate change", "technology"]
)

```

Result : 

```
{'sequence': 'The recent advancements in renewable energy technologies are shaping the future of sustainable development.',
 'labels': ['technology', 'energy', 'environment', 'climate change'],
 'scores': [0.6060120463371277,
  0.3418309986591339,
  0.0490993931889534,
  0.003057534573599696]}
```

Analysis on example 2 : 

The model correctly identifies "technology" as the dominant theme in the sentence, with the highest confidence, followed by "energy," reflecting the focus on renewable energy advancements and their impact on sustainable development.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline('text-generation', model = 'gpt2')
generator("Artificial intelligence is revolutionizing the way we", max_length = 30, num_return_sequences=3)
```

Result : 

```
[{'generated_text': 'Artificial intelligence is revolutionizing the way we look at and interact with our lives. The tech world is getting closer to a point where consumers will need'},
 {'generated_text': 'Artificial intelligence is revolutionizing the way we interact with the world, and is poised to give us unprecedented insights into how our bodies are wired to interact'},
 {'generated_text': 'Artificial intelligence is revolutionizing the way we work — even for a while. Research has demonstrated that the use of artificial intelligence has the potential to become'}]
```

Analysis on example 3 : 

The model generates coherent and contextually relevant continuations of the prompt, showcasing creativity and flexibility in expanding on the theme of artificial intelligence's impact on various aspects of life.


### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "His name is Haris Januar and He lives in Bandung."

ner_results = nlp(example)
print(ner_results)
```

Result : 

```
[{'entity_group': 'PER', 'score': np.float32(0.9976789), 'word': 'Haris Januar', 'start': 11, 'end': 24}, {'entity_group': 'LOC', 'score': np.float32(0.9986707), 'word': 'Bandung', 'start': 40, 'end': 48}]
```

Analysis on example 4 : 

The model accurately identifies "Haris Januar" as a person and "Bandung" as a location, with high confidence scores, demonstrating its effectiveness in named entity recognition for personal and geographical data.

### 5. Example 5 - Question Answering

```
# TODO :
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="What is the capital city of Japan?",
    context="Tokyo is the capital city of Japan. It is one of the most populous cities in the world and is known for its modern architecture, shopping, and entertainment districts. Tokyo is also a major hub for business and culture in Asia.",
)

```

Result : 

```
{'score': 0.975875973701477, 'start': 0, 'end': 5, 'answer': 'Tokyo'}
```

Analysis on example 5 : 

The model accurately extracts "Tokyo" as the answer with a high confidence score, demonstrating its ability to efficiently retrieve specific information from the provided context.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Indonesia is a Southeast Asian country comprising over 17,000 islands, making it the largest archipelago in the world. It is known for its diverse culture, rich history, and stunning natural landscapes, including rainforests, mountains, and beautiful beaches. Jakarta, the capital, is one of the largest urban areas in the world. Indonesia is also home to many ethnic groups and languages, with Bahasa Indonesia serving as the official language. The country is a major player in regional economics and politics and is a member of organizations such as ASEAN and the G20. Despite its economic growth, Indonesia faces challenges like deforestation, natural disasters, and social inequality.
"""
)

```

Result : 

```
[{'summary_text': ' Indonesia is a Southeast Asian country with over 17,000 islands . It is known for its diverse culture, rich history, and stunning natural landscapes . Indonesia is also home to many ethnic groups and languages, with Bahasa Indonesia serving as the official language . The country is a major player'}]

```

Analysis on example 6 :

The summarization model effectively condenses the key points about Indonesia’s geography, culture, and challenges into a concise version, while retaining the core information.

### 7. Example 7 - Translation

```
# TODO :
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("I love spicy food in Bangkok", max_length=40))

```

Result : 

```
[{'translation_text': 'Ich liebe würziges Essen in Bangkok'}]

```

Analysis on example 7 :

The translation model accurately translates the English sentence "I love spicy food in Bangkok" into German, demonstrating its ability to handle casual and context-aware translations.

---

## Analysis on this project

This project demonstrates the versatility of Hugging Face pipelines across various NLP tasks, including sentiment analysis, text generation, named entity recognition, and translation. Each example highlights the models' ability to process and interpret natural language with high accuracy, showcasing their real-world applications in diverse domains.
