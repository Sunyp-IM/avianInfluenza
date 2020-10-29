#Usage:
#python cal.py exampleSequences.fasta

import os
from Bio import SeqIO
import pandas as pd
from itertools import repeat
from multiprocessing import Pool
from Bio.Align.Applications import MafftCommandline
from collections import OrderedDict
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import sys

def encode_int(x0):
    nuclist = ['A', 'T', 'G', 'C', '-']
    char_to_int = dict((c,i) for i,c in enumerate(nuclist))
    intlist = []
    onehotlist = []
    # print 'intlist = :', intlist
    for i in range(x0.shape[1]):
        integer_encoded = [char_to_int[char] for char in x0.iloc[:,i]]
        intlist.append(integer_encoded)
    df_int = pd.DataFrame(intlist).T
    df_int.columns = x0.columns
    return df_int

def mvAmbi(df):
    # Remove ambigious nucleotide acids
    m = ['A', 'C']; df.replace('M',random.SystemRandom().choice(m),inplace=True)
    r = ['A', 'G']; df.replace('R',random.SystemRandom().choice(r),inplace=True)
    w = ['A', 'T']; df.replace('W',random.SystemRandom().choice(w),inplace=True)
    s = ['C', 'G']; df.replace('S',random.SystemRandom().choice(s),inplace=True)
    y = ['C', 'T']; df.replace('Y',random.SystemRandom().choice(y),inplace=True)
    k = ['G', 'T']; df.replace('K',random.SystemRandom().choice(k),inplace=True)
    v = ['A', 'C', 'G']; df.replace('V',random.SystemRandom().choice(v),inplace=True)
    h = ['A', 'C', 'T']; df.replace('H',random.SystemRandom().choice(h),inplace=True)
    d = ['A', 'G', 'T']; df.replace('D',random.SystemRandom().choice(d),inplace=True)
    b = ['C', 'G', 'T']; df.replace('B',random.SystemRandom().choice(b),inplace=True)
    n = ['A', 'G', 'C', 'T']; df.replace('N',random.SystemRandom().choice(n),inplace=True)
    return df

def fasta2dic(infile, path):
    records = SeqIO.parse(infile, 'fasta')
    seqList = []
    for record in records:
        seq = record.seq._data.upper()
        stemList = record.description.split('|')
        if len(stemList) == 2:
            strainName = stemList[0].strip()
            segName = stemList[1].strip()
            seqList.append([strainName, segName, seq])
        elif len(stemList) == 3:
            strainName = stemList[0].strip()
            segName = stemList[1].strip()
            host = stemList[2].strip()
            seqList.append([strainName, segName, host, seq])
        else:
            print('Input error!')
            os._exit(0)
    if len(stemList) == 2:
        seq_df = pd.DataFrame(seqList, columns=['strainName', 'segName', 'seq'])
    elif len(stemList) == 3:
        seq_df = pd.DataFrame(seqList, columns=['strainName', 'segName', 'host', 'seq'])

    # group the sequences by segment names and write them into seperate files in seqPath directory.
    segDic = {'HA': seq_df[(seq_df.segName == 'HA') | (seq_df.segName == '4')], \
              'MP': seq_df[(seq_df.segName == 'MP') | (seq_df.segName == 'M') | (seq_df.segName == '7')], \
              'NA': seq_df[(seq_df.segName == 'NA') | (seq_df.segName == '6')], \
              'NP': seq_df[(seq_df.segName == 'NP') | (seq_df.segName == '5')], \
              'NS': seq_df[(seq_df.segName == 'NS') | (seq_df.segName == '8')], \
              'PA': seq_df[(seq_df.segName == 'PA') | (seq_df.segName == '3')], \
              'PB1': seq_df[(seq_df.segName == 'PB1') | (seq_df.segName == '2')], \
              'PB2': seq_df[(seq_df.segName == 'PB2') | (seq_df.segName == '1')] \
              }
    for key in segDic.keys():
        # print(segDic[key].iloc[0,0],segDic[key].iloc[0,1],segDic[key].iloc[0,2])
        with open(path + key + '.fasta', 'w') as f:
            for i in range(segDic[key].shape[0]):
                if len(stemList) == 2:
                    f.write('>' + segDic[key].iloc[i, 0] + '|' + segDic[key].iloc[i, 1] + '\n')
                    f.write(segDic[key].iloc[i, 2] + '\n')
                else:
                    f.write('>' + segDic[key].iloc[i, 0] + '|' + segDic[key].iloc[i, 1] + '|' + segDic[key].iloc[i, 2] +'\n')
                    f.write(segDic[key].iloc[i, 3] + '\n')
    return segDic

def fasta2df(infile):
    records = SeqIO.parse(infile, 'fasta')
    seqList = []
    for record in records:
        desp = record.description
        # print(desp)
        seq = list(record.seq._data.upper())
        seqList.append([desp] + seq)
        seq_df = pd.DataFrame(seqList)
        seq_df.columns=['strainName']+list(range(1, seq_df.shape[1]))
    return seq_df



def align(infile, outfile):
    mafft_cline = MafftCommandline(input=infile)
    stdout, stderr = mafft_cline()
    with open(outfile, 'w') as file:
        file.write(stdout)

if __name__ == "__main__":
    rootPath = os.getcwd()
    modelPath = rootPath + '/modelSpace/'
    userPath = rootPath + '/userSpace/'
    # exprInput = rootPath + '/' + sys.argv[1]
    exprInput = rootPath + '/' + 'exampleSequences.fasta'
    # exprInput = rootPath + '/' + 'multiStrains.fasta'
    command0 = 'ls -l ' + userPath + ' |grep "^d"|wc -l'
    dirCount = int(os.popen(command0).read())
    if dirCount == 0:
        newDir = userPath + 'job_0/'
        command1 = 'mkdir ' + userPath + 'job_0'
        os.system(command1)
    else:
        newDir = userPath + 'job_' + str(dirCount) + '/'
        command2 = 'mkdir ' + newDir
        os.system(command2)
    command3 = 'mkdir ' + newDir + 'seg'
    os.system(command3)
    command4 = 'mkdir ' + newDir + 'alignedSeg'
    os.system(command4)
    command5 = 'mkdir ' + newDir + 'alignedSegORF'
    os.system(command5)
    command6 = 'mkdir ' + newDir + 'genome'
    os.system(command6)

    df_model = pd.read_csv(modelPath + "model_input_no_ambiguity.csv", index_col=0)
    with open(modelPath + 'modelGenome.fasta', 'w') as f:
        for i in range(df_model.shape[0]):
            f.write('>'+df_model.iloc[i, 0]+'|'+df_model.iloc[i, 1]+'|'+str(df_model.iloc[i, 2])+'\n')
            f.write(''.join(df_model.iloc[i, 3:]) + '\n')

    exprSegPath = newDir + 'seg/'
    exprSegDic = fasta2dic(exprInput, exprSegPath)

    segNameList = ['HA', 'MP', 'PB2', 'PB1', 'PA', 'NP', 'NA', 'NS']
    alignFuncList = []
    p = Pool(10)
    for key in exprSegDic.keys():
        infile = exprSegPath + key + '.fasta'
        outfile = newDir + 'alignedSeg/' + key + '.fasta'
        f = p.apply_async(align, [infile, outfile])
        alignFuncList.append(f)
    for f in alignFuncList:
        f.get()

    # Romove non-coding region
    for key in exprSegDic.keys():
        records = SeqIO.parse(newDir + 'alignedSeg/' + key + '.fasta', 'fasta')
        seqList = []
        for record in records:
            strainName, segName = record.description.split('|')
            seq = list(record.seq._data.upper())
            seqList.append([strainName, segName] + seq)
        exprSegDic[key] = pd.DataFrame(seqList)
        exprSegDic[key].columns = ['strainName', 'segName'] + list(range(1,exprSegDic[key].shape[1]-1))

        # remove 3'-terminal non-coding region
        for i in range(exprSegDic[key].shape[1] - 1, exprSegDic[key].shape[1] - 200, -1):
            # Intercept a sub-dataframe of 3xn from 3'-terminal of the orignial dataframe
            triCode_df = exprSegDic[key].iloc[:, i - 2:i + 1]
            triCode_df.columns = list(range(3))
            oneCode_df = pd.DataFrame([0] * triCode_df.shape[0])
            # Combine each row of the sub-dataframe into a list
            for j in range(oneCode_df.shape[0]):
                oneCode_df.iloc[j, 0] = ''.join(triCode_df.loc[j])

            stopCode_df = oneCode_df[oneCode_df[0].isin(['TAG', 'TGA', 'TAA'])]
            if len(stopCode_df) > len(oneCode_df) * 0.95:
                break
        exprSegDic[key] = exprSegDic[key].iloc[:, :i + 1]

        # Remove 5'-terminal non-coding region
        for i in range(3, 200):
            # Intercept a sub-dataframe of 3xn from 3'-terminal of the orignial dataframe
            triCode_df = exprSegDic[key].iloc[:, i:i + 3]
            triCode_df.columns = list(range(3))
            oneCode_df = pd.DataFrame([0] * triCode_df.shape[0])
            # Combine each row of the sub-dataframe into a list
            for j in range(oneCode_df.shape[0]):
                oneCode_df.iloc[j, 0] = ''.join(triCode_df.loc[j])

            startCode_df = oneCode_df[oneCode_df[0] == 'ATG']
            if len(startCode_df) > len(oneCode_df) * 0.95:
                break
        exprSegDic[key].drop(labels=range(1, i-2), axis=1, inplace=True)

        exprSegDic[key]['segName'] = pd.Series([key] * exprSegDic[key].shape[1])

        # Remove ambiguious nucleotides
        exprSegDic[key] = mvAmbi(exprSegDic[key])

        exprSegDic[key]['segName'] = pd.Series([key] * exprSegDic[key].shape[1])

        # Write to fasta file
        with open(newDir + 'alignedSegORF/' + key + '.fasta', 'w') as f:
            for i in range(exprSegDic[key].shape[0]):
                segName = exprSegDic[key].iloc[i, 0]
                seq = ''.join(exprSegDic[key].iloc[i, 3:])
                f.write('>' + segName + '\n')
                f.write(seq + '\n')

    with open(newDir + 'genome/exprGenome.fasta', 'w') as f:
        seq = ''
        for i in range(exprSegDic['HA'].shape[0]):
            strainName = exprSegDic['HA'].iloc[i, 0]
            host = exprSegDic['HA'].iloc[i, 2]
            ha = ''.join(exprSegDic['HA'].iloc[i, 3:])

            for key in exprSegDic.keys():
                df = exprSegDic[key][(exprSegDic[key].strainName == strainName)]
                if df.shape[0] == 1:
                    seq = seq + ''.join(df.iloc[0, 3:])
                else:
                    print("Error: invalid input sequences!")
            f.write('>' + strainName + '\n')
            f.write(seq + '\n')
            seq = ''

    command7 = 'cat ' + modelPath + 'modelGenome.fasta ' + newDir + 'genome/exprGenome.fasta > ' + newDir + 'genome/conGenome.fasta'
    os.system(command7)

    # Align the combined segment files
    infile = newDir + 'genome/conGenome.fasta'
    outfile = newDir + 'genome/alignedConGenome.fasta'
    align(infile, outfile)

    # convert aligned combined genome to dataFrame, remove nucleotide positions that are not included in model sequences
    df_con = fasta2df(newDir + 'genome/alignedConGenome.fasta')
    n_model = 910
    n_expr = len(df_con) - n_model

    n_expr = len(df_con) - len(df_model)
    col_to_drop = []
    for i in range(df_con.shape[1]):
        if df_con.iloc[:n_model, i].unique().tolist() == ['-']:
            col_to_drop.append(i)
    df_con.drop(col_to_drop, axis=1, inplace=True)

    X = df_con.iloc[-n_expr:, 1:]
    chunksize = 2000
    xlist = []
    for i in range(0, X.shape[1], chunksize):
        xlist.append(X.iloc[:, i:i + chunksize])
    p = Pool(10)
    funclist = []
    for x0 in xlist:
        f = p.apply_async(encode_int, [x0])
        funclist.append(f)
    x_intlist = []
    for f in funclist:
        xint = f.get()
        x_intlist.append(xint)
    x_int = pd.concat(x_intlist, axis=1)

    l = joblib.load(modelPath + 'model_24.m')
    y_pred_proba = l.predict_proba(x_int)[:, 1]
    strainName = df_con.iloc[-n_expr:, 0]
    strainName.index = range(strainName.shape[0])
    result = pd.concat([strainName, pd.Series(y_pred_proba)], axis=1)
    result.columns = ['strainName', 'Infectivity']
    result.to_excel(newDir + 'result.xls')
