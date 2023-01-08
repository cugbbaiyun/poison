def q():
  for i in range(2):
    yield i

g = q()
print(next(g))
print(next(g))
if not g.__next__():
   print("test")
