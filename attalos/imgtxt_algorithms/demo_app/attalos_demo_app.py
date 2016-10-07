from flask import Flask,request, render_template, send_from_directory
import numpy as np
import h5py
import gzip
from attalos.util.wordvectors.glove import GloveWrapper
from os.path import basename
app = Flask(__name__)
algo1={'name': 'negsamp fixed 2048,1024,200',
       'word_vec_file': '/path/to/word/vector/file', 
       'hdf5':'/path/to/hdf5/output/of/infer',
       'norm_vectors': False}
id_lookup = {'dict': None, 'txt': '/path/to/file/mapping/id/to/url'}


def tags_2_vec(tags, w2v_model=None, normalize=False):
    good_tags = [tag.lower() for tag in tags if tag.lower() in w2v_model]
    if len(good_tags) == 0:
        raise KeyError('Tags not found: {}'.format(tags))
    else:
        word_vectors = []
        for tag in good_tags:
            word_vector = w2v_model[tag]
            if normalize:
                word_vector /= np.linalg.norm(word_vector)
            word_vectors.append(word_vector)
        output = np.sum(word_vectors, axis=0)
        if normalize:
            output /= np.linalg.norm(output)
        return output.astype(np.float32)


def get_divs(image_urls, scores):
    output = ""
    for url,score in zip(image_urls, scores):
        output += '''<div class="grid-item">
      <img src="%s" /> %0.3f
    </div>\n''' % (url, score)
    return output


def get_url_from_filename(filename):
    global id_lookup
    if id_lookup['dict'] is None:
        print('Loading id lookup')
        id_lookup['dict'] = {}
        fname = id_lookup['txt']
        if fname.endswith('gz'):
            open_fn = gzip.open(fname)
        else:
            open_fn = open(fname)
        for line in open_fn:
            line = line.strip().split('\t')
            bname = basename(line[5])
            id_lookup['dict'][bname] = line[5]
    bname = basename(filename)
    return id_lookup['dict'][bname]


def get_search_results(query, algo, k=15, normalize=True):
    global word_lookup_filename
    if 'init' not in algo:
        print('Initializing from: {}'.format(algo['hdf5']))
        hdf5_file = h5py.File(algo['hdf5'])
        if True:
            algo['img_data'] = np.zeros(hdf5_file['preds'].shape, dtype=np.float32)
            hdf5_file['preds'].read_direct(algo['img_data'])
        else:
            algo['img_data'] = hdf5_file['preds']
        if algo['norm_vectors']:
            algo['img_data'] = (algo['img_data'].T / np.linalg.norm(algo['img_data'], axis=1)).T
        algo['ids'] = hdf5_file['ids']
        algo['w2v'] = GloveWrapper.load(algo['word_vec_file'])
        algo['init'] = True

    query_vector = tags_2_vec(query.strip().lower().split(','), algo['w2v'])
    indices = np.argpartition(np.dot(algo['img_data'], query_vector), -1*k)[-1*k:]
    print(max(np.dot(algo['img_data'], query_vector)))
    indices = indices[::-1]
    score_urls = [(np.dot(algo['img_data'][index],query_vector),
                   get_url_from_filename(algo['ids'][index])) for index in indices]
    score_urls = sorted(score_urls, reverse=True)
    print(score_urls)
    scores = [score for score, url in score_urls]
    urls = [url for score, url in score_urls]
    html = get_divs(urls, scores)
    return html

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)

@app.route('/', methods=['GET', 'POST', 'PUT'])
def landing_page():
    global algo1
    global algo2
    default_text = 'Search...'
    query = request.args.get('query', default_text)
    if query == default_text:
        return render_template('index.html', search_bar=default_text)

    # Add case where word not found
    try:
        result1 = get_search_results(query, algo1)
        return render_template('index.html',
                               search_bar=query,
                               algo1_name=algo1['name'],
                               algo1_results=result1)
    except KeyError:
        raise
        return render_template('index.html', search_bar=default_text)



