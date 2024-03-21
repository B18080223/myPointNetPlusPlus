import os
# 循环读取txt
def readData(filePath):
    dataPathSet = []
    fileNames=os.listdir(filePath)
    for fileName in fileNames:
        if fileName.endswith('.txt'):
            dataPathSet.append(filePath+"\\"+fileName)
            # dataPathSet.append(fileName.split('.')[0])
    return dataPathSet


if __name__ == '__main__':
    filePath="E:\Desktop\cloud"
    print(readData(filePath))