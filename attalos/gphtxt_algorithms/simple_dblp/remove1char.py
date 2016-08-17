import sys

filename=sys.argv[1]

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
