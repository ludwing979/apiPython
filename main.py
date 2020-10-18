import os
from app import app
from FaceRecognition import fun, getData
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


# POST - just get the image and metadata
@app.route('/RequestImageWithMetadata', methods=['POST'])
def post():
	if request.method == 'POST':
		# check if the post request has the file part
		request_data = request.form['some_text']
		print(request_data)
		if 'avatar_img' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['avatar_img']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			result = fun(request_data)
			return result



@app.route('/ch2', methods=['POST'])
def data():
	if request.method == 'POST':
		request_data1 = request.form['n1']
		request_data2 = request.form['some_text']
		if request_data1 == '1':
			return getData(request_data2)
		else:
			return "Tu información no ha sido compartida"











if __name__ == "__main__":
    app.run(host= '0.0.0.0')