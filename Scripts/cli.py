import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Script for training an AutoEncoder')

    parser.add_argument('dataset',
                        metavar='dataset',
                        type=str,
                        help='dataset name')
                       
    parser.add_argument('seed',
                        metavar='seed',
                        type=int,
                        help='the initialization seed')
                                                                     
    parser.add_argument('metric',
                        metavar='dist',
                        choices=['euclidean',
                                 'manhattan'],
                        type=str,
                        help='distance metric to use')

    parser.add_argument('-reg',
                        metavar='regularizer',
                        choices=['reg1', 
                                 'reg2', 
                                 'reg1log', 
                                 'reg2log', 
                                 'sammon', 
                                 'noreg'],
                        default='noreg',
                        type=str,
                        help='the regularization formulation')

    parser.add_argument('-coeff',
                        metavar='reg_coeff',
                        type=float,
                        help='the regularization coefficient')
    
    parser.add_argument('-msecoeff',
                        metavar='mse_coeff',
                        type=float,
                        help='the regularization coefficient')
    
#     parser.add_argument('-clip',
#                         metavar='clip',
#                         choices=['norm',
#                                  'val',
#                                  None],
#                         default=None,
#                         help='the clip function to use')

    args = parser.parse_args()

    if args.reg != 'noreg' and args.coeff is None:
        parser.error('Reg coeff needs to be specified')

    print(args)
    return args
