#-------------------------------------------------------------
#import require library
#-------------------------------------------------------------

from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

#-------------------------------------------------------------
# load model
#-------------------------------------------------------------

model = YOLO("best.onnx")

#-------------------------------------------------------------
#create folder for result
#-------------------------------------------------------------

UPLOAD_FOLDER = "uploads"
IMAGE_RESULT = "static/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_RESULT, exist_ok=True)

#-------------------------------------------------------------

video_path = None      #vedio path none becuse live stream no need to save vedio

#-------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global video_path

    #load img , vedio
    #-------------------------------------------------------------

    file = request.files["file"]

    #-------------------------------------------------------------
    #to find the type of file (vedio/img) 
    #if vedio .mp4 if img .jpg
    #-------------------------------------------------------------

    ext = file.filename.rsplit(".",1)[1].lower()

    #-------------------------------------------------------------
    #genrate random unique id 
    #-------------------------------------------------------------

    filename = str(uuid.uuid4()) + "." + ext

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    image_ext = ["jpg","jpeg","png","bmp"]
    video_ext = ["mp4","avi","mov","mkv"]

    if ext in image_ext:

        results = model.predict(filepath, conf=0.3)

        save_path = os.path.join(IMAGE_RESULT, filename)

        results[0].save(filename=save_path)

        return render_template(
            "index.html",
            image=save_path
        )

    elif ext in video_ext:

        video_path = filepath

        return render_template(
            "index.html",
            stream=True
        )

    return "Unsupported File"


def generate_frames():

    global video_path

    cap = cv2.VideoCapture(video_path)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(frame, conf=0.3)

        annotated = results[0].plot()

        #-------------------------------------------------------------
        #simple array use to Display 
        #-------------------------------------------------------------
        _, buffer = cv2.imencode('.jpg', annotated)

        #-------------------------------------------------------------
        #convert array into bytes Browsers pc understand bytes 
        #-------------------------------------------------------------

        frame = buffer.tobytes()

        #-------------------------------------------------------------
        #yield used to puse while return break function the code
        #------------------------------------------------------------- 
        yield (b'--frame\r\n'     #it tells the browser A new image is starting
               b'Content-Type: image/jpeg\r\n\r\n' +  #it tell browser The following data is a JPEG image
               frame +                   #continue to next frame
               b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=True)