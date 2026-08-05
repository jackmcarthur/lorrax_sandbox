import re, numpy as np
def parse_sigma_hp(path):
    blocks=[]; ik=None; kcrys=None
    for line in open(path):
        s=line.strip()
        m=re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys=(float(m.group(1)),float(m.group(2)),float(m.group(3))); ik=int(m.group(4)); continue
        if ik is None: continue
        p=s.split()
        if len(p)>=15 and p[0].isdigit():
            n=int(p[0])
            if not any(b.get('ik')==ik for b in blocks): blocks.append({'kcrys':kcrys,'ik':ik,'bands':{}})
            blocks[-1]['bands'][n]={'X':float(p[3]),'SXmX':float(p[4]),'CH':float(p[5]),'Sig':float(p[6]),
                'Vxc':float(p[7]),'Eqp0':float(p[8]),'Eqp1':float(p[9]),'CHp':float(p[10]),'Sigp':float(p[11]),
                'Cor':float(p[4])+float(p[5]),'Corp':float(p[4])+float(p[10])}
        elif len(p)==11 and p[0].isdigit():
            n=int(p[0])
            if not any(b.get('ik')==ik for b in blocks): blocks.append({'kcrys':kcrys,'ik':ik,'bands':{}})
            blocks[-1]['bands'][n]={'X':float(p[3]),'SXmX':float(p[4]),'CH':float(p[5]),'Sig':float(p[6]),
                'Vxc':float(p[7]),'Eqp0':float(p[8]),'Eqp1':float(p[9]),'CHp':float(p[5]),'Sigp':float(p[6]),
                'Cor':float(p[4])+float(p[5]),'Corp':float(p[4])+float(p[5])}
    return blocks

import sys
path=sys.argv[1]
bl=parse_sigma_hp(path)
print("path:",path,"nblocks:",len(bl))
for b in bl:
    if tuple(round(x,3) for x in b['kcrys'])==(0.0,0.0,0.0):
        print("Gamma ik=",b['ik'])
        for n in sorted(b['bands']):
            d=b['bands'][n]
            print(f"  n={n:2d} X={d['X']:8.3f} SX-X={d['SXmX']:8.3f} CH={d['CH']:8.3f} Cor={d['Cor']:8.3f} Sig={d['Sig']:8.3f} Vxc={d['Vxc']:8.3f} Eqp0={d['Eqp0']:8.3f} Eqp1={d['Eqp1']:8.3f}")
        break
