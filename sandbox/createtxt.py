def dblp2text(items, contexts):

  for idx in xrange(items):
     title=items[idx]
     authorlist=contexts[idx]
     for author in authorlist:
	print author+" "+title
