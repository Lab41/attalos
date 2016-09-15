
import json

# Helper to get the datas
def json_load(fp):
    with open(fp) as infile:
        return json.load(infile)
    
def load_output_json(fp):
    """A function to load DenseCap predictions. The function assumes the input
    json is of the form:
        {
            opt : 
                {
                    image_size: 720,
                    ...
                },
            results : 
                [
                    {
                        boxes : [[x, y, w, h], [x, y, w, h], ...],
                        captions : [<word>, ...],
                        img_name : <file path>,
                        scores : [<score>, <score>, ...],
                    }
                ]

        }
    Will return a dictionary matching the output of `load_groundtruth_json(fp)`
    
    Parameters
    -----------
    fp : string
        path to DenseCap output results
        
    Returns
    ---------
    dictionary
    """
    parsed = json_load(fp)
    transformed_img_size = parsed['opt']['image_size']
    out = {}
    for result in parsed['results']:
        img_name = result['img_name']
        items = []
        for idx in xrange(len(result['scores'])):
            names = [result['captions'][idx]]
            
            # got this order form densecap box_utils
            x, y, w, h = result['boxes'][idx]
            
            score = result['scores'][idx]
            items.append(dict(names=names, x=x, y=y, h=h, w=w, score=score))
        out[img_name] = items
    return out
            
        
        
    
def load_groundtruth_json(fp):
    """A function to load DenseCap training data. The function assumes 
    the input json is of the form:
        [
            {
                id: <number>,
                objects : 
                    [
                        {
                            x: <number>, 
                            y : <number>,
                            h : <number>, 
                            w : <number>,
                            id : <number>,
                            names: ['list', 'of', 'names']
                        }
                    ]
                }
            }
        ]
        
    Will return a dictionary using the image name (id + .jpg) and bounding box 
    information

    Parameters
    -----------
    fp : string
        path to a json file

    Returns
    --------
    dictionary
    """
    parsed = json_load(fp)
    out = {}
    for item in parsed:
        # the formatted input only holds a number that is later turned into a 
        # filename within densecap code
        src_img = "{0}.jpg".format(item['id'])
        out[src_img] = item['objects']
    return out    


def load_formatted_objects_json(fp):
    """A function to load formatted object data data. The function assumes 
    the input json is of the form:
        [
            {
                id: <number>,
                regions : 
                    [
                        {
                            x: <number>, 
                            y : <number>,
                            height : <number>, 
                            width : <number>,
                            id : <number>,
                            phrase: "somthing cool",
                            image : <image id>,
                        }
                    ]
                }
            }
        ]
        
    Will return a dictionary using the image name (id + .jpg) and bounding box 
    information

    Parameters
    -----------
    fp : string
        path to a json file

    Returns
    --------
    dictionary
    """
    parsed = json_load(fp)
    out = {}
    for item in parsed:
        # the formatted input only holds a number that is later turned into a 
        # filename within densecap code
        src_img = "{0}.jpg".format(item['id'])
        regions = item['regions']
        out_regions = []
        for region in regions:
            formatted = dict(x=region['x'], y=region['y'], h=region['height'], 
                             w=region['width'], names=[region['phrase']])
            out_regions.append(formatted)
        out[src_img] = out_regions
    return out