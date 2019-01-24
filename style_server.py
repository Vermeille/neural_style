from flask import Flask, request
import requests
import style
import base64
import io
from PIL import Image
from types import SimpleNamespace
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '.'


def img_to_data_url(img):
    with io.BytesIO() as f:
        img = Image.open(img)
        img.save(f, 'jpeg')
        f.seek(0)
        return 'data:image/jpg;base64,' + base64.b64encode(f.read()).decode()


def fname(f):
    return os.path.basename(f).split('.')[0]


stylizer = style.ArtisticStyleOptimizer(device='cuda')
@app.route('/go', methods=['GET', 'POST'])
def go():
    rargs = request.form
    args = SimpleNamespace()

    args.size = int(rargs['size'])
    args.scale = float(rargs['scale'])
    args.ratio = float(rargs['ratio'])
    args.content_layers = [rargs['content_layer']]
    args.preserve_colors = rargs.get('preserve_colors', 'off') == 'on'

    fn = 'content/' + str(int(time.time() * 100)) + '.png'
    request.files['content'].save(fn)
    args.content = fn

    fn = 'style/' + str(int(time.time() * 100)) + '.png'
    request.files['style'].save(fn)
    args.style = fn

    args.out = 'result/' + fname(args.content) + '_' + fname(args.style) + '.png'
    style.go(args, stylizer)

    return '<img src="' + img_to_data_url(args.out) +'" />'


@app.route('/', methods=['GET'])
def index():
    return """
<!doctype html>
<html>
    <head>
        <link href="https://unpkg.com/filepond/dist/filepond.css" rel="stylesheet">
    </head>
    <body>
        <script src="https://unpkg.com/filepond/dist/filepond.js"></script>
        <h1>Neural Style</h1>
        <form id="upload" action="/go" method="POST" enctype="multipart/form-data">
            <input
                type="file"
                class="filepond"
                name="content"
                placeholder="content URL"
                required/>
            <br/>
            <input
                type="file"
                class="filepond"
                name="style"
                placeholder="style URL"
                required/>
            <br/>
            <input
                type="number"
                name="ratio"
                placeholder="Loss ratio"
                value="300"/>
            <br/>
            <input
                type="number"
                name="scale"
                placeholder="Style Scale"
                value="1"/>
            <br/>
            Content shape fidelity:
            <select name="content_layer">
                <option value="relu3_2">high</option>
                <option value="relu4_2">medium</option>
                <option value="relu5_2">low</option>
            </select>

            Result size:
            <select name="size">
                <option value="128">128px</option>
                <option value="256">256px</option>
                <option value="512">512px</option>
                <option value="720">720px</option>
                <option value="1024">1024px</option>
            </select>
            <br/>
            Preserve Colors <input type="checkbox" name="preserve_colors" />
            <br/>
            <input type="submit"/>
        </form>
        <script>
            FilePond.parse(document.body);
        </script>
    </body>
    <script>
window.upload.onsubmit = (e) => {
};
    </script>
</html>
    """

if __name__ == '__main__':
    try:
        os.mkdir('content')
        os.mkdir('style')
        os.mkdir('result')
    except:
        pass
    app.run(host='0.0.0.0', port=8000)
