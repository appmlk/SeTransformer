sbt_leng = 300
code_leng = 200
nl_leng = 30


def get_Data(bath, path):
    sbts = []
    sbtposs = []
    sbtsizes = []
    sbtmasks = []

    codes = []
    codesizes = []
    codemasks = []

    inputNL = []
    outputNL = []
    nlopuputLeng = []
    f = open('data/' + path + '/sbt', 'r', encoding='utf-8')
    SBTs = f.readlines()
    f.close()
    for temp in SBTs:
        sbt = [int(w) for w in temp.strip().split()]
        sbt = [2] + sbt + [3]
        if len(sbt) > sbt_leng:
            sbt = sbt[0:sbt_leng]
        if len(sbt) % 2 == 0:
            ls = len(sbt) // 2
        else:
            ls = len(sbt) // 2 + 1
        sbtsizes.append(ls)
        sbtmasks.append([1] * ls + [0] * (sbt_leng//2 - ls))
        while len(sbt) < sbt_leng:
            sbt.append(0)
        sbts.append(sbt)

    f = open('data/' + path + '/ids', 'r', encoding='utf-8')
    SBTs = f.readlines()
    f.close()
    for temp in SBTs:
        sbt = [int(w) for w in temp.strip().split()]
        sbt = [0] + sbt + [0]
        if len(sbt) > sbt_leng:
            sbt = sbt[0:sbt_leng]
        while len(sbt) < sbt_leng:
            sbt.append(0)
        sbtposs.append(sbt)

    f = open('data/' + path + '/code', 'r', encoding='utf-8')
    CODEs = f.readlines()
    f.close()
    for temp in CODEs:
        code = [int(w) for w in temp.strip().split()]
        code = [30001] + code + [30002]
        if len(code) > code_leng:
            code = code[0:code_leng]
        if len(code) % 2 == 0:
            lc = len(code) // 2
        else:
            lc = len(code) // 2 + 1
        codesizes.append(lc)
        codemasks.append([1] * lc + [0] * (code_leng//2 - lc))
        while len(code) < code_leng:
            code.append(0)
        codes.append(code)

    if path == 'train':
        f = open('data/' + path + '/nl', 'r', encoding='utf-8')
        NLs = f.readlines()
        f.close()
        for temp in NLs:
            nl = [int(w) for w in temp.strip().split()]
            nl = [2] + nl + [3]
            inp = nl[0:-1]
            outp = nl[1:len(nl)]
            if len(inp) > nl_leng:
                inp = inp[0:nl_leng]
                outp = outp[0:nl_leng]
            nlopuputLeng.append(len(inp))
            while len(inp) < nl_leng:
                inp.append(0)
                outp.append(0)
            inputNL.append(inp)
            outputNL.append(outp)
    else:
        f = open('data/' + path + '/nl.char', 'r', encoding='utf-8')
        NLs = f.readlines()
        f.close()
        for temp in NLs:
            nl = [w for w in temp.strip().split()]
            if len(nl) > nl_leng:
                nl = nl[:nl_leng]
            outputNL.append(nl)
            nlopuputLeng.append(len(nl))
            nl = [2]
            for i in range(nl_leng - 1):
                nl.append(0)
            inputNL.append(nl)

    bathsbt = []
    bathsbtpos = []
    bathsbtsize = []
    bathsbtmask = []
    bathcode = []
    bathcodesize = []
    bathcodemask = []
    bathinputNL = []
    bathoutputNL = []
    bathnloutputLeng = []
    start = 0
    while start < len(codes):
        end = min(start + bath, len(codes))
        bathsbt.append(sbts[start:end])
        bathsbtpos.append(sbtposs[start:end])
        bathsbtsize.append(sbtsizes[start:end])
        bathsbtmask.append(sbtmasks[start:end])

        bathcode.append(codes[start:end])
        bathcodesize.append(codesizes[start:end])
        bathcodemask.append(codemasks[start:end])

        bathinputNL.append(inputNL[start:end])
        bathoutputNL.append(outputNL[start:end])

        bathnloutputLeng.append(nlopuputLeng[start:end])
        start += bath
    return bathsbt, bathsbtpos, bathsbtsize, bathsbtmask, bathcode, bathcodesize, bathcodemask, bathinputNL, bathoutputNL, bathnloutputLeng
