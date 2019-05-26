import copy
# 1. 逻辑： 循环长度 循环位置
# 2. 先匹配索引 再根据索引判断字符串，填入列表
# 3. 去重
'''查询相同字段'''
def search_public_str(str1, str2, length):
    result_set = []
    for long in range(1, length):
        for start_index in range(length):
            try:
                search = str1[start_index:start_index+long]
                if search in str2:
                    result_set.append(search)
            except:
                break
    result_set = list(set(result_set))
    return result_set

'''列表去重'''
def quchong(result_set):
    pub_str_set = copy.deepcopy(result_set)
    for o_data in result_set:
        for p_data in result_set:
            if p_data in o_data and p_data != o_data:
                try:
                    pub_str_set.remove(p_data)
                except:
                    continue
    return pub_str_set

if __name__ == '__main__':
    str1 = 'hellow wordxx'
    str2 = 'wordsobeautifulxxx'
    length = len(str1)

    result_set = search_public_str(str1, str2, length)
    print('全部结果',result_set)
    result = quchong(result_set)
    print('短结果',result)
