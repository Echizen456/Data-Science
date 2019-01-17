import urllib.request
f = urllib.request.urlopen('http://www.douban.com/')
print(f.read(300).decode('utf-8'))
print(f.info())
