## Application to demonstrate image search

# Usage: 
A dockerfile is provided. All that is required is a conda environment with Flask installed


```bash
# From the top level of the attlos repository

PYTHONPATH=$PWD/ FLASK_APP=attalos/imgtxt_algorithms/demo_app/attalos_demo_app.py flask run --host=0.0.0.0
```
