from flask import Flask, render_template, request
from transformers import pipeline

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi model parafrase
paraphraser = pipeline("text2text-generation", model="t5-base")

# Halaman utama dengan formulir input teks
@app.route("/", methods=["GET", "POST"])
def home():
    paraphrased_text = ""
    if request.method == "POST":
        original_text = request.form["original_text"]
        result = paraphraser(original_text, max_length=50, num_return_sequences=1)
        paraphrased_text = result[0]['generated_text']
    return render_template("index.html", paraphrased_text=paraphrased_text)

if __name__ == "__main__":
    app.run(debug=True)