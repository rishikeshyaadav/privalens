from flask import Flask, render_template, request, jsonify
import os
import base64

app = Flask(__name__)

# DATABASE CONFIG
DB_FOLDER = "faces_db"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

@app.route('/')
def index():
    return render_template('register.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    if not data or 'image' not in data or 'name' not in data:
        return jsonify({'message': 'Missing data'}), 400

    image_data = data['image']
    user_name = data['name'].strip()

    if user_name == "":
        return jsonify({'message': 'Please enter a name'}), 400

    try:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        filename = f"{user_name}.jpg"
        filepath = os.path.join(DB_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(binary_data)
        print(f"✅ Registered: {filename}")
        return jsonify({'message': f'Success! {user_name} registered.'}), 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'message': 'Failed to save image'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)