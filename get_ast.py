import sys
sys.setrecursionlimit(1000000)
class NODE:
    def __init__(self, no):
        self.node = no
        self.child = []

def dfs(root, sbt, pos, i):
    de = sbt[i]
    ned = NODE(de)
    root.child.append(ned)
    ps = int(pos[i])
    i+=1
    flag = 0
    while True:
        if i >=len(pos):
            break
        x = int(pos[i])
        if x == ps:
            flag = 1
            if sbt[i] != de:
                nw = NODE(sbt[i])
                ned.child.append(nw)
            i+=1
            continue
        if flag == 1:
            break
        i = dfs(ned, sbt, pos, i)
    return i

def dfs_ast(root, father, nodes, fa, num):
    nodes.append(root.node)
    father.append(str(fa))
    wc = num
    for ch in root.child:
        num = dfs_ast(ch, father, nodes, wc, num+1)
    return num

def getast(sbt, pos):
    root = NODE('root')
    dfs(root, sbt, pos, 0)
    father = []
    nodes = []
    dfs_ast(root.child[0], father, nodes, 0, 1)
    fas = ''
    for c in father:
        fas+=c+' '
    fas +='\n'

    nos = ''
    for c in nodes:
        nos += c + ' '
    nos += '\n'
    return nos, fas


def get_path(path):
    f = open('temp2/'+path+'/sbt', 'r', encoding='utf-8')
    SBT = f.readlines()
    f.close()

    f = open('temp2/' + path + '/pos', 'r', encoding='utf-8')
    POS = f.readlines()
    f.close()

    f1 = open('data/' + path + '/ast', 'w', encoding='utf-8')
    f2 = open('data/' + path + '/father', 'w', encoding='utf-8')

    for sbt, pos in zip(SBT, POS):
        ast, fathr = getast(sbt.split(), pos.split())
        f1.write(ast)
        f2.write(fathr)
    f1.close()
    f2.close()

get_path('test')
get_path('train')
get_path('valid')