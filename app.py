import sys
import base64
if sys.platform == 'linux':
	import Xlib.threaded
from flask import Flask, render_template, Response, request, jsonify
from camera_desktop import Camera
from flask_socketio import SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# async_mode = 'eventlet'
socketio = SocketIO(app)

@app.route('/')
def index():
	return render_template('index.html')


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def background_vid_broadcast(camera=Camera()):
	while True:
		socketio.sleep(0.1)
		frame, data = camera.get_frame()
		jpg_as_text = base64.b64encode(frame).decode('ascii')
		socketio.emit(
			'video_feed',
			{
				'img_frame': jpg_as_text,
				'data':data
			},
                broadcast=True,
                namespace='/test',
                ignore_queue=True)


@socketio.on('start_stream', namespace='/test')
def video_feed(message):
	# global thread
    # with thread_lock:
    #     if thread is None:
    vid_thread = socketio.start_background_task(background_vid_broadcast)


@socketio.on('connect')
def test_connect():
	print("client connected")

@socketio.on('test_event', namespace='/test')
def handle_message(message):
	print('received message: ' + str(message))

# @app.route('/video_feed')
# def video_feed():
# 	return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False)
	# app.run(host='0.0.0.0', threaded=True)