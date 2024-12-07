from flask import Flask, request, render_template, jsonify, json, redirect, url_for
import selectorlib
import requests
import analyze
import os.path
from werkzeug.utils import safe_join

app = Flask(__name__)
extractor = selectorlib.Extractor.from_yaml_file('selectors.yml')

def scrape(url):
    headers = {
        'authority': 'www.amazon.com',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
    
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = extractor.extract(r.text)
        return data
    else:
        return {'error': f'Failed to retrieve data from Amazon. Status code: {r.status_code}'}
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape_data():
    url = request.form.get('url')
    if not url or 'amazon.com' not in url:
        return jsonify({'error': 'Please provide a valid Amazon URL'}), 400
    
    data = scrape(url)
    reviews = []
    
    if data.get("reviews"):
        reviews = analyze.analyse(data["reviews"])
            
    name = url[23:].split("/", 1)[0]
    filename = f"{name}.json"
    
    if reviews:
        with open(f"data/{filename}", 'w') as f:
            json.dump(reviews, f)
            
    return redirect(url_for('show_reviews', filename=filename))

@app.route('/reviews')
def show_reviews():
    filename = request.args.get('filename')
    filepath = safe_join('data', filename)
    reviews = []
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            reviews = json.load(f)
    return render_template('reviews_clustered.html', clusters=reviews)
    
if __name__ == '__main__':
    app.run(debug=True)