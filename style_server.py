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

def wget(url, nm):
    r = requests.get(url)
    print(r.status_code)
    with io.BytesIO(r.content) as f:
        im = Image.open(f)
        im.save(nm)


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
    rargs = request.args
    args = SimpleNamespace()

    args.size = int(rargs['size'])
    args.scale = float(rargs['scale'])
    args.ratio = float(rargs['ratio'])
    args.preserve_colors = rargs.get('preserve_colors', 'off')

    args.content = rargs['content']
    if args.content.startswith('http'):
        fn = 'content/' + str(int(time.time() * 100)) + '.png'
        wget(args.content, fn)
        args.content = fn

    args.style = rargs['style']
    if args.style.startswith('http'):
        fn = 'style/' + str(int(time.time() * 100)) + '.png'
        wget(args.style, fn)
        args.style = fn

    args.out = 'result/' + fname(args.content) + '_' + fname(args.style) + '.png'
    print(args.__dict__)
    style.go(args, stylizer)

    return '<img src="' + img_to_data_url(args.out) +'" />'


@app.route('/', methods=['GET'])
def index():
    return """
<!doctype html>
<html>
    <body>
        <h1>Neural Style</h1>
        <form id="upload" action="/go" method="GET">
            <input
                type="text"
                name="content"
                placeholder="content URL"
                required/>
            <br/>
            <input
                type="text"
                name="style"
                placeholder="style URL"
                required/>
            <br/>
            <input
                type="number"
                name="ratio"
                placeholder="Loss ratio"
                value="10"/>
            <br/>
            <input
                type="number"
                name="scale"
                placeholder="Style Scale"
                value="1"/>
            <br/>
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
