import numpy as np

def find_corners():
    ret = []
    tick = np.array([-1,1])
    for x in tick:
        for y in tick:
            for z in tick:
                s = (x, y, z)
                ret.append(s)
    return np.array(ret)

def find_edges():
    ret = np.array([
        (0, -1, -1),
        (0, -1, 1),
        (0, 1, -1),
        (0, 1, 1),
        (-1, 0, -1),
        (-1, 0, 1),
        (1, 0, -1),
        (1, 0, 1),
        (-1, -1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (1, 1, 0)
    ])
    return ret

def write2file(fp):
    for c in ret:
        x, y, z = c
        x1, y1, z1 = c + 1
        d = 7 * x1 + 2 * y1 + 1 * z1
        idx.append(d)
        s = '(%+d %+d %+d) - (%d %d %d) : %d' % (x,y,z, x1,y1,z1, d)
        print(s)
        fp.write(s)
        fp.write('\n')

fp = open('positions.md', 'w')

idx = []

print('# Corner cubelets')
fp.write('# Corner cubelets\n')
ret = find_corners()
write2file(fp)

print()
fp.write('\n')

print('# Edge cubelets')
fp.write('# Edge cubelets\n')
ret = find_edges()
write2file(fp)

print()
fp.write('\n')
s = str(sorted(idx))
print(s)
fp.write(s)
fp.write('\n')

fp.close()
