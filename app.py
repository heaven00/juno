import os
from flask import Flask, jsonify, send_from_directory, request, make_response
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import numpy as np



app = Flask(__name__)

DEFAULT_WEIGHTS = 'models/weights.npy'
UPLOAD_PATH = 'img/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
ROOT_PATH = os.getcwd()

def get_solver():
    NetClass = load_model('ResidualGRUNet')
    net = NetClass(compute_grad=False)
    net.load(DEFAULT_WEIGHTS)
    return Solver(net)

def process_image(imgs):
    ims = []
    for im in imgs:
        if im.size[0] != im.size[1]:
            min_size = min(im.size)
            im = im.resize((min_size, min_size), Image.ANTIALIAS)
        im.thumbnail((127, 127), Image.ANTIALIAS)
        try:
            bg = Image.new("RGB", (127, 127), (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
        except:
            bg = im
        im_matrix = np.array(bg)
        ims.append([im_matrix.transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return ims

def generate_image_map():
    img_map = {}
    for index, item in enumerate(os.listdir('img')):
        img_map[index] = item
    return img_map

SOLVER = get_solver()
IMGS = generate_image_map()

@app.route("/")
def hello():
    return "Hello, I am Juno!."

@app.route('/list')
def get_model_list():
    return jsonify(IMGS)

@app.route('/get/<id>')
def get_model(id):
    try:
        model_name = 'img/' + IMGS[int(id)]
        response = make_response(send_from_directory(ROOT_PATH, model_name))
        response.headers["Content-Disposition"] = "attachment; filename=test.obj"
        return response
    except KeyError:
        return jsonify({'message': 'Invalid Object ID'})

@app.route('/add/<name>', methods=['POST'])
def image_to_model(name):
    if request.method == 'POST':
        imagedata = []
        print(len(list(request.files.values())))
        print(list(request.files.keys()))
        for imagefile in request.files.values():
            imagedata.append(Image.open(BytesIO(imagefile.read())))
    if len(imagedata) == 0:
        return jsonify({'message': 'No Image Found.'})
    voxel_prediction, _ = SOLVER.test_output(process_image(imagedata))
    voxel_obj_name = 'img/' + name + '.obj'
    voxel2obj(voxel_obj_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)
    response = make_response(send_from_directory(ROOT_PATH,voxel_obj_name))
    response.headers["Content-Disposition"] = "attachment; filename=test.obj"
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
