def intersection(list1, list2):
    setintersection = []
    list1index = []
    list2index = []
    for word in list1:
        if word in list2:
            setintersection += [word]
            list1index += [list(list1).index(word)]
            list2index += [list(list2).index(word)]
    return setintersection, list1index, list2index

# Union will keep the order of the original training list
def union(list1, list2):
    setunion = []
    list1index = []
    list2index = []
    for word in list(list1)+list(list2):
        if word in setunion:
            continue
        else:
            setunion += [word]
            if word in list1:
                list1index += [list(list1).index(word)]
            else:
                list1index += [-1]
            if word in list2:
                list2index += [list(list2).index(word)]
            else:
                list2index += [-1]
    return setunion, list1index, list2index

def difference(list1, list2):
    setdiff = []
    list1index = []
    for word in list1:
        if word not in list2:
            setdiff+=[word]
            list1index += [list(list1).index(word)]
    return setdiff, list1index

def replaceword(dictionary, wordsearch, wordreplace):
    found = False
    if wordreplace in dictionary:
        print "Warning: {} already in dictionary. Nothing changed".format(wordreplace)
        return found
    for i,word in enumerate(dictionary):
        if word == wordsearch:
            dictionary[i]=wordreplace
            found = True
    if found:
        print 'Found and replaced {} at index {} with {}'.format(wordsearch, i, wordreplace)
    else:
        print 'Did not find {} in provided dictionary'.format(wordsearch)
    return found