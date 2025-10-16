from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Python API is running!"

@app.route('/double', methods=['POST'])
def double_number():
    data = request.get_json()
    num = data.get('number', 0)
    return jsonify({'result': num * 2})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
