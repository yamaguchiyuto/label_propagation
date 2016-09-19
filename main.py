import sys
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from label_propagation import HMN,LGC,PARW,OMNI,CAMLP

def read_graphfile(f):
    graph_data = np.genfromtxt(f, delimiter=' ', dtype=int)
    row = np.hstack([graph_data[:,0], graph_data[:,1]])
    col = np.hstack([graph_data[:,1], graph_data[:,0]])
    max_nid = np.max(row)
    return csr_matrix((np.ones(len(row)), (row,col)), shape=(max_nid+1,max_nid+1))

def read_labelfile(f):
    label_data = np.genfromtxt(f, delimiter=' ', dtype=int)
    return label_data[:,0],label_data[:,1]

p = argparse.ArgumentParser()
subparsers = p.add_subparsers(help='sub-command help', title='subcommands', dest='subparser_name')

hmn_p = subparsers.add_parser('hmn', help='HMN')
hmn_p.add_argument("-g", "--graphfile", help="input graph file", type=argparse.FileType('r'), required=True)
hmn_p.add_argument("-l", "--labelfile", help="input label file", type=argparse.FileType('r'), required=True)
hmn_p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
lgc_p = subparsers.add_parser('lgc', help='LGC')
lgc_p.add_argument("-g", "--graphfile", help="input graph file", type=argparse.FileType('r'), required=True)
lgc_p.add_argument("-l", "--labelfile", help="input label file", type=argparse.FileType('r'), required=True)
lgc_p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
lgc_p.add_argument("--alpha", help="alpha (default=0.99)", type=float, nargs='?', default=0.99)
parw_p = subparsers.add_parser('parw', help='PARW')
parw_p.add_argument("-g", "--graphfile", help="input graph file", type=argparse.FileType('r'), required=True)
parw_p.add_argument("-l", "--labelfile", help="input label file", type=argparse.FileType('r'), required=True)
parw_p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
parw_p.add_argument("--lamb", help="lambda (default=1.0)", type=float, nargs='?', default=1.0)
omni_p = subparsers.add_parser('omni', help='OMNI')
omni_p.add_argument("-g", "--graphfile", help="input graph file", type=argparse.FileType('r'), required=True)
omni_p.add_argument("-l", "--labelfile", help="input label file", type=argparse.FileType('r'), required=True)
omni_p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
omni_p.add_argument("--lamb", help="lambda (default=1.0)", type=float, nargs='?', default=1.0)
camlp_p = subparsers.add_parser('camlp', help='CAMLP')
camlp_p.add_argument("-g", "--graphfile", help="input graph file", type=argparse.FileType('r'), required=True)
camlp_p.add_argument("-l", "--labelfile", help="input label file", type=argparse.FileType('r'), required=True)
camlp_p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
camlp_p.add_argument("--beta", help="beta (default=0.1)", type=float, nargs='?', default=0.1)
camlp_p.add_argument("--modulationfile", help="modulation matrix file (default: use identity)", type=argparse.FileType('r'), nargs='?', default=None)

args = p.parse_args()

G = read_graphfile(args.graphfile).tolil()
x,y = read_labelfile(args.labelfile)

if args.subparser_name == 'hmn':
    clf = HMN(graph=G)
elif args.subparser_name == 'lgc':
    clf = LGC(graph=G,alpha=args.alpha)
elif args.subparser_name == 'parw':
    clf = PARW(graph=G,lamb=args.lamb)
elif args.subparser_name == 'omni':
    clf = OMNI(graph=G, lamb=args.lamb)
elif args.subparser_name == 'camlp':
    H = np.genfromtxt(args.modulationfile, delimiter=' ')
    clf = CAMLP(graph=G, beta=args.beta, H=H)

clf.fit(x,y)
predicted = clf.predict_proba(np.arange(G.shape[0]))

print >>args.outfile, '"Node ID","Predicted label ID",%s' % ','.join(['"Prob %s"'%v for v in range(predicted.shape[1])])
for i in range(predicted.shape[0]):
    print >>args.outfile, "%s,%s,%s" % (i,predicted[i].argmax(),','.join(map(str,predicted[i])))
