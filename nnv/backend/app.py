from flask import Flask, render_template
from flask import jsonify, request

import neuralNetworkSchema

app = Flask(__name__)

@app.route('/stick', methods=['GET', 'POST'])
def stick():
    if request.method == 'POST':
        result = request.form['string1'] + request.form['string2']
        return render_template('index.html', result=result)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()