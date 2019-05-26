str_1 = '1274214'

start_index = ''
end_index = ''
all_datas = []
length = len(str_1)

for start_index in range(length): # 0 6
    for end_index in range(start_index, length+1): # 0 6
        data = str_1[start_index:end_index]
        all_datas.append(data)

all_datas = set(all_datas)
print(all_datas)