# Natural Language Processing with Hugging Face Transformers
**Generative AI Guided Project on Cognitive Class by IBM**

**Name**: Nabilah Putri Wijaya  
**Universitas Semarang**

## My TODO:

### 1. Example 1 - Sentiment Analysis
```python
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")
Result:

json
Copy code
[{'label': 'POSITIVE', 'score': 0.9959210157394409}]
Analysis on Example 1:
The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.

2. Example 2 - Topic Classification
python
Copy code
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.",
    candidate_labels=["science", "pet", "machine learning"],
)
Result:

json
Copy code
{'sequence': 'Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.',
 'labels': ['pet', 'machine learning', 'science'],
 'scores': [0.9174826145172119, 0.048576705157756805, 0.03394068405032158]}
Analysis on Example 2:
The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

3. Example 3 and 3.5 - Text Generator
python
Copy code
# TODO :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "This cooking will make you",
    max_length=30, # you can change this
    num_return_sequences=2, # and this too
)
Result:

json
Copy code
[{'generated_text': 'This cooking will make you even richer. I used to work too little, I thought it was kind of ridiculous to take it that far. I was'},
 {'generated_text': 'This cooking will make you feel alive for hours every afternoon. It would also help keep your children in school throughout the day.\n\n\nOne of'}]
Analysis on Example 3:
The text generation model produces coherent and imaginative continuations of a cooking-themed prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

4. Example 4 - Name Entity Recognition (NER)
python
Copy code
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Arifian, I am an AI Technical Mentor at Infinite Learning, Batam Island")
Result:

json
Copy code
[{'entity_group': 'PER',
  'score': np.float32(0.9978566),
  'word': 'Arifian',
  'start': 11,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.7615841),
  'word': 'AI',
  'start': 28,
  'end': 30},
 {'entity_group': 'ORG',
  'score': np.float32(0.9623977),
  'word': 'Infinite Learning',
  'start': 51,
  'end': 68},
 {'entity_group': 'LOC',
  'score': np.float32(0.9913697),
  'word': 'Batam Island',
  'start': 70,
  'end': 82}]
Analysis on Example 4:
The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

5. Example 5 - Question Answering
python
Copy code
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What four-legged animal sometimes comes inside the house and likes to sleep?"
context = "Four-legged animal that sometimes comes inside the house and likes to sleep is a cat"
qa_model(question = question, context = context)
Result:

json
Copy code
{'score': 0.6314472556114197, 'start': 79, 'end': 84, 'answer': 'a cat'}
Analysis on Example 5:
The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

6. Example 6 - Text Summarization
python
Copy code
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
    Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan sistem komputer untuk belajar dari data tanpa diprogram secara eksplisit. 1  Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu. Penerapannya luas, mulai dari rekomendasi produk hingga diagnosis medis, mengubah cara kita berinteraksi dengan teknologi. 
    """
)
Result:

json
Copy code
[{'summary_text': ' Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit . Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu .'}]
Analysis on Example 6:
The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

7. Example 7 - Translation
python
Copy code
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
Result:

json
Copy code
[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]
Analysis on Example 7:
The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

Analysis on this Project
This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
