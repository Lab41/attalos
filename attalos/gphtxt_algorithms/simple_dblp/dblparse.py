import sys
'''
This class creates a parse of the DBLP dataset. It will go through quickly and extract relevant fields.
It will also be able to aggregate fields together, provided you know what the delimiting field to be.
Look for examples in the main function at the bottom.
'''


class dblparser:

    def __init__(self, dataset, maxiter=1000):
        dbname=dataset
        self.db = open(dataset,'r')
        self.MAXITER=maxiter

    def getnext(self, item='<title>', num2get=1):
        '''
        items = getnext( item='<title>', num2get=1 )

        input:
          - item can be '<title>', '<author>', etc.
          - num2get >= 1
        '''

        items=[]
        for num in xrange(num2get):
            for iter in xrange(self.MAXITER):
                line = self.db.readline()

                if not line:
                    print "End of file reached while looking for next title"
                    break

                if len(line) > 10:
                    if line[0:len(item)] == item:
                        items.append(line)
                        break

            if iter==self.MAXITER-1:
                print "ERROR, max iteration reached before finding title"

        return items

    def striplist(self, thelist):

        newlist = []
        for line in thelist:
            bol= line.find('>')
            eol= line.find('<', 3)
            newlist.append( line[bol+1:eol] )
        return newlist

    def stripitem(self, line):
        bol = line.find('>')
        eol = line.find('<',3)
        return line[bol+1:eol] 

    def list2file(self, thelist, filename):

        fid = open(filename,'w')
        for line in thelist:
            fid.write('%s\n' % line)

    def getnextpaper(self, num2get=1, fields=['<author>','<title>'], lastfield='<title>', strip=True):
        '''
        Assume that the title is the last
        fields
        '''
        items=[]
        contexts=[]
        for num in xrange(num2get):
            itemstring=''
            contextlist = []

            for iter in xrange(self.MAXITER):
                line = self.db.readline()

                if not line:
                    break

                if len(line) < 10:
                    continue

                for field in fields:
                    
                    if line[0:len(lastfield)]==lastfield:
                        if strip:
                            itemstring=self.stripitem(line)
                        else:
                              itemstring = line
                        break

                    if line[0:len(field)]==field:
                        if strip:
                            contextlist.append(self.stripitem(line))
                        else:
                            contextlist.append(line)
                        break

                if itemstring:
                    items.append(itemstring)
                    contexts.append(contextlist)
                    break

        return items,contexts

    def paper2lines(self, items, contexts):

        for idx in xrange(0,len(items)):
            title=items[idx]
            authorlist=contexts[idx]
            if authorlist:
                for author in authorlist:
                    print author+" "+title

def remove1char(filename):
    fid = open(filename,'r')
    while True:
        line = fid.readline()
        printline=""
        if not line:
            break
        words = line.split(' ')
        for word in words:
            if len(word) > 1:
                printline+=word+" "

        sys.stdout.write(printline)

if __name__ == "__main__":
    
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        sys.stdout.write("USAGE: python dblparse.py <num-records> <optional - file-name>")
    if len(sys.argv) == 2:
        filename = 'dblp.xml'
    else:
        filename = sys.argv[2]
    num2get = int(sys.argv[1])
    
    dbp = dblparser(filename)
    items,contexts = dbp.getnextpaper(num2get=num2get)
    dbp.paper2lines(items,contexts)
    
# dbp = dblparser('dblp-2015-03-02.xml')
# dbp = dblparser('dblp.xml')
# titles = dbp.getnext(num2get=10)
# authors= dbp.getnext(item='<author>', num2get=100000)
# items,contexts = dbp.getnextpaper(num2get=3000000)
# dbp.paper2lines(items,contexts)
