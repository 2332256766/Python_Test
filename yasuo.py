import copy,re
'''解压压缩'''
# 纯字母字符串   # str = 'aabcccccddddd'  # zip_str = 'a2bc5d5'

# 先切片 # 以字母变化时的索引为分界
# 再匹配连接的长度
# 长度大于2则改变字符串
# 数字 拼接
def zip(data):
    # 压缩
    zip_list,Pointer_1,Pointer_2 = [],0,0
    for index,info in enumerate(data+'$'): # 遍历索引 信息
        if index ==0:
            continue
        if data[index-1] != info: # 如果前一个数据不等于当前数据 # 索引为0时无之前数据
            Pointer_1 = Pointer_2 # 赋值前指针（索引开始）
            Pointer_2 = index # 赋值后指针（索引结束）
            zip_list.append(data[Pointer_1:Pointer_2])

    result = lambda data:data[0] + str(len(data)) if len(data)!=1 else data[0] # 匿名函数 返回data长度
    zip_list = [result(data) for data in zip_list] # 写法一 # 学过
    # zip_list = list(map(result, zip_list)) # 写法二 # 对列表子元素操作函数
    return ''.join(zip_list)

# 根据数字切割 匹配到数字，乘上之前的字母
# 字符串相乘
# 字符串 拼接
def unzip(input_data):
    # 解压
    unzip_list,Pointer_1,Pointer_2 = [],0,0
    for index,info in enumerate(input_data): # 遍历索引 信息
        # 捕捉字符串
        try:      # 如果转化数字跳过
            int(info)
            continue
        except:   # 如果是字符串
            Pointer_1 = index+1
            # 索引后一个可能为数字
            try:
                int(input_data[Pointer_1])  # 如果为数字
                Pointer_2 = Pointer_1
                while True:
                    try:
                        Pointer_2 += 1
                        int(input_data[Pointer_2])
                    except:
                        num = int(input_data[Pointer_1: Pointer_2])
                        unzip_list.append(info * num)
                        break
            except:
                unzip_list.append(info)     # 如果不为数字 添加
    return ''.join(unzip_list)

if __name__ == '__main__':
    input_str = 'aabcccccdddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
    print('输入:',input_str)
    zip_output= zip(input_str)
    print('压缩：',zip_output)
    result = unzip(zip_output)
    print('解压缩：',result)