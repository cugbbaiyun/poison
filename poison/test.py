with open('../lstur/datas/train/news_poisoned.tsv') as f:
  line = f.readline()
  line = line.split('\t')
  print(line)
  print(len(line))

