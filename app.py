from flask import Flask, request, render_template, jsonify, json
import selectorlib
import requests
import analyze
import os.path

app = Flask(__name__)
extractor = selectorlib.Extractor.from_yaml_file('selectors.yml')

def scrape(url):
    headers = {
        'authority': 'www.amazon.com',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
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
    data = [data["reviews"][i]["content"][:-10] for i in range(len(data["reviews"]))]
    name = url[23:] # remove https://www.amazon.com/
    name = name.split("/", 1)[0]
    print(name)
    filename = f"data/{name}.json"
    if not os.path.isfile(filename):
        analyze.run(data, filename)
    data
    with open(filename) as json_file:
        data = json.load(json_file)
    return jsonify(data)
    # return render_template("result.html", data = data)
    

if __name__ == '__main__':
    app.run(debug=True)
