import re

f = open("io.c","r")
t = f.read()
b = re.findall('case ([A-Z_]+):\s*strcpy\(\s*buf\s*,\s*\"([^\"]*)', t)
a = re.findall('case ([A-Z_]+):\s*strncpy\(\s*label\s*,\s*\"([^\"]*)', t)


d1 = {}
for k,v in a:
    d1[k] = v



for k,v in b:
    print "'"+v.strip()+"': '" + d1[k].strip().lower() + "',"
    
    

e = re.findall('init_field\(.*"(.{4})".*"(.*)"',t)

for k,v in e:
    print "'"+v.strip()+"': '" + k.strip().lower() + "',"
