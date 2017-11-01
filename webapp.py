

from flask import Flask, request, g, render_template, redirect, url_for

from io import BytesIO, StringIO
import requests
import urllib.request as urllib
from urllib.parse import urlparse
import base64
import os

from PIL import Image
from firefly import Client

app = Flask(__name__)
# app.config['SECRET_KEY'] = os.environ['SECRET_KEY']

# find the project name from the env and use that to construct
# the endpoint for the API.
NAME = os.getenv("PROJECT", "background-removal")
API_ENDPOINT = "https://{}--api.rorocloud.io".format(NAME)

api_req = Client(API_ENDPOINT)

@app.route('/predict', methods=['POST'])
def predict():

	if (request.method == 'POST'):

		img_url = request.form['image_url']
		validator = urlparse(img_url)

		if(validator.scheme == 'http' or validator.scheme == 'https'):

			try:
				img = Image.open(BytesIO(urllib.urlopen(img_url).read()))
			except:
				message = "Could not retreive image, try another link"
				return render_template('index.html', message=message)

			# Send back the result image to the client
			img_req = BytesIO()
			img.save(img_req, 'JPEG')
			img_req.seek(0)

			# Call the Background Removal API
			img_resp = api_req.predict(image_file=img_req, format='jpg')

			# Prepare response for display
			encoded_image = base64.b64encode(img_resp.read()).decode('ascii')

			return render_template('index.html', encoded_image=encoded_image)
		else:
			# Invalid URL handling
			message = "Not a valid URL, try again"
			return render_template('index.html', message=message)
	else: 
		return redirect(url_for('homepage'))

@app.route('/works')
def works():
    return render_template('works.html')

@app.route('/')
def homepage():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()