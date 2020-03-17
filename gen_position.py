import numpy as np

def find_cubelets(nZero):
    ret = []
    axis = np.array([-1,0,1])
    for x in axis:
        for y in axis:
            for z in axis:
                coord = np.array([x, y, z])
                if coord[coord == 0].size == nZero:
                    s = '(%d, %d, %d)' % (x, y, z)
                    ret.append(s)
    return ret

fp = open('positions.md', 'w')

print('# Corner cubelets')
fp.write('# Corner cubelets\n')
ret = find_cubelets(0)
print(ret)
for c in ret:
    fp.write('%s\n' % str(c))

print()
fp.write('\n')

print('# Edge cubelets')
fp.write('# Edge cubelets\n')
ret = find_cubelets(1)
print(ret)
for c in ret:
    fp.write('%s\n' % str(c))

fp.close()
