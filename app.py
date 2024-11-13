from flask import Flask, render_template, request
from transformers import pipeline
import os

# Set the cache directory to a writable location
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"

app = Flask(__name__)

# Load the paraphraser model
paraphraser = pipeline("text2text-generation", model="t5-base")

@app.route("/", methods=["GET", "POST"])
def home():
    paraphrased_text = ""
    if request.method == "POST":
        original_text = request.form["original_text"]
        # Use the paraphrasing model with some parameters
        result = paraphraser(original_text, max_length=50, num_return_sequences=1)
        paraphrased_text = result[0]['generated_text']
    return render_template("index.html", paraphrased_text=paraphrased_text)

if __name__ == "__main__":
    # Use host='0.0.0.0' for compatibility with platforms like Vercel
    app.run(debug=True, host="0.0.0.0")
