# import the Flask class from the flask module
from flask import Flask, redirect, render_template, request, flash
from werkzeug.utils import secure_filename
from seam_carving import run
import os

# create the application object
app = Flask(__name__)
app.secret_key = "super secret key"
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = os.getcwd()+"/static"
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# use decorators to link the function to a url
# Some code for accepting uploaded images adapted from:
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
@app.route("/", methods=["GET", "POST"])
def home():
    errors = ""
    if request.method == "POST":
        # Check if the file is included in the POST
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Check if the user uploaded an empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If the file is good, then save it in the UPLOAD_FOLDER
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        horizontal_seams = None
        vertical_seams = None
        # Check if the integer input is as expected
        try:
            horizontal_seams = int(request.form["horizontal_seams"])
            vertical_seams = int(request.form["vertical_seams"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["horizontal_seams"])

        # Call seam_carving to do the work
        name = "/static/"+filename
        qualified_name = os.getcwd()+name
        name, vis_name = run(qualified_name, horizontal_seams, vertical_seams)
        return render_template('result.html', \
            name=name, vis_name=vis_name, horizontal_seams=horizontal_seams, \
                                        vertical_seams=vertical_seams) 

    return render_template('index.html')  

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)