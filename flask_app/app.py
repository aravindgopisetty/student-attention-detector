from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def index():
    try:
        with open("attention_score.txt", "r") as f:
            pct = float(f.read().strip())
    except:
        pct = 0.0
    return render_template("index.html", pct=pct)

if __name__ == "__main__":
    print("→ [app.py] Launching Flask server at http://127.0.0.1:5000 …")
    app.run(host="127.0.0.1", port=5000, debug=True)

