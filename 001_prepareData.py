import numpy as np
import copy
import os,sys,datetime,time,gzip
import math
import random
import argparse

# print information line
def infoLine(message, infoType="info"):
    
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)


def readData(infile, file_type='count'):  
    geneList = []
    cellIdList = []
    cell2groupHash = {}
    group2cellHash = {}
    group2codeHash = {}

    expHash  = {}
    hasLowNumGroup = False

    f1st = True
    with open(infile, "rt") as fi:
        for line in fi:
            row = line.rstrip().split("\t")
            if f1st:
                f1st = False
                geneList = row[2:]
                continue

            cellId = row[0]
            group  = row[1]
            cellIdList.append(cellId)
            cell2groupHash[cellId] = group

            expHash[cellId] = {}

            if group not in group2cellHash:
                group2cellHash[group] = {}
            group2cellHash[group][cellId] = ""

            row_float = [float(k) for k in row[2:]]

            if file_type == 'normalized':
                norm_row = row_float 
            else:
                lib = sum(row_float)
                if lib > 0:
                    norm_row = [k * 10000.0 / lib for k in row_float]
                else:
                    norm_row = row_float

            for k in range(len(geneList)):
                expHash[cellId][geneList[k]] = norm_row[k]

    groupList = sorted(list(group2cellHash.keys()))
    for index in range(len(groupList)):
        group = groupList[index]
        group2codeHash[group] = str(index)
        if len(group2cellHash[group]) < 30:
            hasLowNumGroup = True
            print("Cell number from " + group + " is lower than 30")

    if hasLowNumGroup:
        infoLine("Please remove clusters or cell types with < 30 cells", "ERROR")

    return expHash, geneList, cellIdList, cell2groupHash, group2cellHash, group2codeHash
    

def filterGenes(expHash, geneList, group2cellHash, file_type, threshold=0.01):
    if file_type == 'normalized':
        # threshold = 0.1  # for log-normalized
        nonzero_exprs = [expHash[cellId][gene] for cellId in expHash for gene in expHash[cellId] if expHash[cellId][gene] > 0]
        threshold = np.percentile(nonzero_exprs, 25) 

    goodGeneList = []
    for geneName in geneList:
        vList_all = [expHash[cellId][geneName] for cellId in expHash]

        if max(vList_all) < threshold:
            continue

        isGroupSpecialGene = False
        for group in group2cellHash:
            vList_group = [expHash[cellId][geneName] for cellId in group2cellHash[group]]
            if np.percentile(vList_group, 80) > threshold:
                isGroupSpecialGene = True
                break

        if isGroupSpecialGene:
            goodGeneList.append(geneName)

    return sorted(goodGeneList)
#

def saveGeneExpressionByGroup(expHash, cell2groupHash, goodGeneList, outDIR):
    expByGroupHash = {}
    
    for cellId in expHash:
        group = cell2groupHash[cellId]
        if group not in expByGroupHash:
            expByGroupHash[group] = {}
        #
        
        for geneName in goodGeneList:
            if geneName not in expByGroupHash[group]:
                expByGroupHash[group][geneName] = []
            #
            expByGroupHash[group][geneName].append( expHash[cellId][geneName] )
        #
    #
    
    npy = outDIR + "/expression_by_group_gene.npy"
    np.save( npy, expByGroupHash )
#

def generateReference(expHash, goodGeneList, refNum, outDIR, file_type):
    geneHash = {}
    # calculate mean and std
    for geneName in goodGeneList:
        if file_type == 'count':
            vlist = [ math.log2(expHash[cellId][geneName] + 1.0) for cellId in expHash ] 
        else:
            vlist = [ expHash[cellId][geneName] for cellId in expHash ]
        geneHash[geneName] = {}
        
        geneHash[geneName]["mean"] = np.mean(vlist)
        geneHash[geneName]["std"]  = np.std(vlist)
    #
        
    # generate references
    refData = np.array([ np.absolute(np.random.normal(geneHash[geneName]["mean"], geneHash[geneName]["std"], refNum)) for geneName in goodGeneList ])
    
    # set decimal5
    decimal = 5
    addData = np.round( np.random.randint(1, 9, size = refData.shape) * 0.0001 + np.random.randint(1, 9, size = refData.shape) * 0.00001, decimal)
    refData = np.round( np.round( refData, 3 ) + addData, decimal )
    
    # transpose
    refData = list(np.transpose(refData))
    
    with open(outDIR + "/universal.dat", "wt") as fo:
        for sv in refData:
            sv = [ "{:.5f}".format(k) for k in sv ]
            fo.write( "universal\t9999\t" + "|".join( sv ) + "\n" )
    #
#

def formatGeneExpression(cell2groupHash, group2codeHash, expHash, goodGeneList, file_type):
    for cellId in expHash:
        group = cell2groupHash[cellId]
        code  = group2codeHash[group]
        if file_type == 'count':
            vlist = [ math.log2(1.0 + expHash[cellId][geneName] ) for geneName in goodGeneList ]
        else:
            vlist = [ expHash[cellId][geneName]  for geneName in goodGeneList ]
            
        vlist = [ "{:.3f}".format(k) for k in vlist ]
        expHash[cellId] = group + "\t" + code + "\t" + "|".join( vlist )
    #
#

def preparePoolData(expHash, group2codeHash, group2cellHash, outDIR):
    cellIdList = sorted( list( expHash.keys() ) )
    # for pool data
    with open(outDIR + "/pool.dat", "wt" ) as fo:
        for cellId in cellIdList:
            fo.write( expHash[cellId] + "\n" )
    #
    
    with open(outDIR + "/sampleList.pool.dat", "wt" ) as fo:
        fo.write( "\n".join( cellIdList ) + "\n" )
    #
#

def prepareTrainTestData(expHash, cell2groupHash, group2codeHash, group2cellHash, outDIR, random_seed=42):
    random.seed(random_seed)
    list_train = []
    list_test  = []
    
    for group in group2cellHash:
        dlist = [ expHash[cellId] for cellId in group2cellHash[group] ]
        random.shuffle(dlist)
        
        if len(dlist) > 50:
            list_train.extend( dlist[:int(len(dlist)*0.9) ] )
            list_test.extend(  dlist[ int(len(dlist)*0.9):] )
        else:
            list_train.extend( dlist[:-10]  )
            list_test.extend(  dlist[-10:]  )
        #
    #

    random.seed(random_seed)
    random.shuffle(list_train)
    random.shuffle(list_test)
    
    with open(outDIR + "/train.dat", "wt" ) as fo:
        fo.write( "\n".join( list_train ) + "\n" )
    #
    
    with open(outDIR + "/test.dat", "wt" ) as fo:
        fo.write( "\n".join( list_test ) + "\n" )
    #
    
    return len( list_train ), len(list_test)
#

def saveCellInfoData(group2codeHash, outDIR):
    with open(outDIR + "/cellCode.dat", "wt" ) as fo:
        for group in group2codeHash:
            fo.write( group2codeHash[group] + "\t" + group + "\n" )
    #
#

def saveGeneInfoData(goodGeneList, outDIR):
    with open(outDIR + "/orderedGeneList.dat", "wt" ) as fo:
        fo.write( "\n".join(goodGeneList) + "\n" )
    #
#

def saveMetaData(refNum, count_train, count_test, goodGeneList, expHash, group2codeHash, outDIR):
    with open( outDIR + "/meta.dat", "wt" ) as fo:
        fo.write("class"     + "\t" + str(len(group2codeHash)) + "\n" )
        fo.write("gene"      + "\t" + str(len(goodGeneList)) + "\n" )
        fo.write("reference" + "\t" + str(refNum) + "\n" )
        fo.write("pool"      + "\t" + str(len(expHash)) + "\n" )
        fo.write("train"     + "\t" + str(count_train) + "\n" )
        fo.write("test"      + "\t" + str(count_test) + "\n" )
    #
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 1.0 \nDescription: prepare datasets for neural network",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="infile",    type=str,   required=True,  help="Path of input data")
    parser.add_argument("-o", dest="outDIR",    type=str,   required=True,  help="The directory of output data")
    parser.add_argument("-t", dest="file_type", choices=["count", "normalized"], default="count", help="Input data type: 'count' or 'normalized'")

    
    args=parser.parse_args()

    infile       = args.infile # 
    outDIR       = args.outDIR #
    file_type    = args.file_type
    refNum       = 256 # number of random references
    
    os.system("mkdir -p " + outDIR)
    infoLine("Read data ... ")
    expHash, geneList, cellIdList, cell2groupHash, group2cellHash, group2codeHash = readData(infile, file_type)
    
    
    infoLine("Filter informative genes ... ")
    goodGeneList = filterGenes(expHash, geneList, group2cellHash, file_type)
    print(str(len(goodGeneList)), "out of ", str(len(geneList)), "genes will be used for further analysis" )
    
    
    infoLine("Save gene expression data ... ")
    saveGeneExpressionByGroup(expHash, cell2groupHash, goodGeneList, outDIR)
    
    
    infoLine("Generate references ... ")
    generateReference(expHash, goodGeneList, refNum, outDIR, file_type)
    
    
    infoLine("Fromat gene expression data ... ")
    formatGeneExpression(cell2groupHash, group2codeHash, expHash, goodGeneList, file_type)
    
    
    infoLine("Prepare pool data ... ")
    preparePoolData(expHash, group2codeHash, group2cellHash, outDIR)
    
    
    infoLine("Prepare train and test data ... ")
    count_train, count_test = prepareTrainTestData(expHash, cell2groupHash, group2codeHash, group2cellHash, outDIR)
    
    
    infoLine("Save cell information ... ")
    saveCellInfoData(group2codeHash, outDIR)
    
    
    infoLine("Save gene information ... ")
    saveGeneInfoData(goodGeneList, outDIR)
    
    
    infoLine("Save meta ... ")
    saveMetaData(refNum, count_train, count_test, goodGeneList, expHash, group2codeHash, outDIR)
    
    infoLine("Done!")
#
