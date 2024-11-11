from transformers import pipeline

# Inisialisasi model parafrase
paraphraser = pipeline("text2text-generation", model="t5-base")

# Teks asli
original_text = "Parafrase AI dapat membantu mengubah teks secara otomatis menjadi kalimat yang berbeda tanpa mengubah makna."

# Menghasilkan parafrase
paraphrased_text = paraphraser(original_text, max_length=50, num_return_sequences=1)

# Menampilkan hasil parafrase
print("Original Text:", original_text)
print("Paraphrased Text:", paraphrased_text[0]['generated_text'])