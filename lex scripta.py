from transformers import pipeline

# Initialize the paraphrasing model using the "text2text-generation" pipeline and T5 model
paraphraser = pipeline("text2text-generation", model="t5-base")

# Original text to be paraphrased
original_text = "Parafrase AI dapat membantu mengubah teks secara otomatis menjadi kalimat yang berbeda tanpa mengubah makna."

# Generate paraphrase
paraphrased_text = paraphraser(original_text, max_length=50, num_return_sequences=1)

# Display the original and paraphrased text
print("Original Text:", original_text)
print("Paraphrased Text:", paraphrased_text[0]['generated_text'])
