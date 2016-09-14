

def print_list_to_columns(words, items_per_row=5):
    """function to print a list of words in column format
    
    Parameters
    ----------
    words : list
        list of words or terms to print
    items_per_row : int
        number of words in a row
    """
    row = []
    width = max(map(len, words)) + 2 
    for idx, word in enumerate(words):
        if (idx + 1) % items_per_row == 0:
            print("".join(word.ljust(width) for word in row))
            row = []
        row.append(word)
    # append one last time just in case
    if len(row) > 0:
        print("".join(word.ljust(width) for word in row))
        
        
def sort_results(boxes):
    """Returns the top n boxes based on score given DenseCap 
    results.json output
    
    Parameters
    ----------
    boxes : dictionary
        output from load_output_json
    n : integer
        number of boxes to return
    
    Returns
    -------
    sorted dictionary
    """
    return sorted(results[k], key=lambda x : x['score'], reverse=True)