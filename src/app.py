from flask import Flask, render_template, request, redirect, url_for
import os
from src.object_detection import perform_detection  # Import your detection function


app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../web_app/templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../web_app/static')))


# Ensure the uploads folder exists
UPLOAD_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform detection on the uploaded image
        result = perform_detection(file_path)

        # Pass the result to the result page
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)