import numpy as np


def GuassianKernel(sigma, dim):  #产生高斯核
    """
    :param sigma: 标准差
    :param dim: 高斯核的维度（必须是个奇数）
    :return: 返回高斯核
    """
    temp = [t - (dim // 2) for t in range(dim)]  # 生成二维高斯的x与y  列表中使用for循环
    assistant = []
    for i in range(dim):
        assistant.append(temp)   #生成三维数组-1 0 1
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + assistant.T ** 2) / temp)  # 二维高斯公式 assitant.T表示转置
    return result


def convolve(kernel, img, padding, strides):
    """
    :param kernel: 输入的核函数
    :param img: 输入的图片
    :param padding: 需要填充的位置
    :param strides: 高斯核移动的步长
    :return: 返回卷积的结果
    """
    result = None
    kernel_size = kernel.shape
    img_size = img.shape
    if len(img_size) == 3: # 三通道图片就对每个通道分别卷积
        channel = []
        for i in range(img_size[-1]): #对每个通道进行填充
            pad_img = np.pad(img[:,:,i],((padding[0],padding[1]),(padding[2],padding[3])),'constant') #进行padding操作
            temp = []
            for j in range (0,img_size[0],strides[1]):   #进行卷积操作 高度
                temp.append([])
                for k in range(0,img_size[1], strides[0]):  #宽度
                    val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                    k * strides[0]:k * strides[0] + kernel_size[1]]).sum()  #依次进行卷积操作
                    temp[-1].append(val)
            channel.append(np.array(temp))
        channel = tuple(channel) #元组
        result = np.dstack(channel) #将列表中的数组沿深度方向进行拼接
    elif len(img_size) == 2:
        channel = []
        pad_img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3])),
                         'constant')  # pad是填充函数 边界处卷积需要对边界外根据高斯核大小填0
        for j in range (0,img_size[0],strides[1]):
            channel.append([])
            for k in range(0,img_size[1],strides[0]):
                val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                k * strides[0]:k * strides[0] + kernel_size[1]]).sum()  # 卷积的定义 相当于用高斯核做加权和
                channel[-1].append(val)
        result = np.array(channel)
    return result


def undersampling (img,step = 2):
    """
    下采样
    :param img: 输入图片
    :param step: 降采样步长 默认为2（）缩小两倍
    :return: 返回降采样结果
    """
    return img[::step,::step]


def getDoG(img,n,sigma0,S=None,O=None):
    """
    :param img: 输入图像
    :param n: 有几层用于提出特征
    :param sigma0: 输入的sigma
    :param S: 金字塔每层有几张gauss滤波后的图像
    :param O: 金字塔有几层
    :return: 返回差分高斯金字塔和高斯金字塔
    """
    if S is None:
        S = n + 3 # 至少有4张 （第一张和最后一张高斯金字塔无法提取特征，差分以后的第一张和最后一张也无法提取特征）
    if O is None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3  # 计算最大可以计算多少层 O=log2（min(img长，img宽））-3
    k = 2 ** (1.0/n)
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)] for o in range(O)]  # 每一层 sigma按照 k^1/s * sigama0  排列 下一层的sigma都要比上一层sigma大两倍
    sample = [undersampling(img, 1 << o) for o in range(O)]  # 降采样取图片作为该层的输入
    Guass_Pyramid = []
    for i in range(O):
        Guass_Pyramid.append([]) #声明二维空数组
        for j in range(S):
            dim = int (6*sigma[i][j]+1) # 通常，图像处理只需要计算（6*sigma+1）*(6*sigma+1)的矩阵就可以保证相关像素影像
            if dim % 2 == 0: #防止高斯核不是奇数
                dim += 1
                dim += 1
            Guass_Pyramid[-1].append(convolve(GuassianKernel(sigma[i][j], dim), sample[i],
                                              [dim // 2, dim // 2, dim // 2, dim // 2],[1, 1]))  # 在第i层添加第j张 经过高斯卷积的 该图片四周扩展 5//2=2 用于高斯卷积
    DoG_Pyramid = [[Guass_Pyramid[o][s + 1] - Guass_Pyramid[o][s] for s in range(S - 1)] for o in range(O)]  #每一层中 上一张减去下一张得到高斯核
    return DoG_Pyramid,Guass_Pyramid,O  #返回高斯金字塔和高斯差分金字塔


def Extrema(img, img_prev, img_next, threshhold):
    extremas = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i][j]
            eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]),
                                    max(0, j - 1):min(j + 2, img_prev.shape[1])]
            eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]),
                               max(0, j - 1):min(j + 2, img.shape[1])]
            eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]),
                                    max(0, j - 1):min(j + 2, img_next.shape[1])]
            # 阈值化，在高斯差分金字塔中找极值
            # 如果某点大于阈值，并且，比周围8个点、上下18个点共26个点都大或者都小，则认为是关键点
            if np.abs(val) > threshhold and \
                    ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (
                            val >= eight_neiborhood_next).all())
                     or (val < 0 and (val <= eight_neiborhood_prev).all() and (
                                    val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):
                extremas.append((i, j))
    return extremas


def getExtremas(DoG_Pyramid=None, T=None, S=None):
    threshold = 0.5 * T / S
    extremas = []
    for i in range(len(DoG_Pyramid)):
        octave = DoG_Pyramid[i]
        oc = []
        for j in range(1, len(octave) - 1):
            points = Extrema(img=octave[j], img_prev=octave[j - 1], img_next=octave[j + 1], threshhold=threshold)
            oc.append(points)
        extremas.append(oc)
    return extremas