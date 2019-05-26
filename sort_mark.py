mark_list = ['沃尔玛']*4+['特斯拉']*10+['比特币']*6+['贸易战']*11
mark_set = set(mark_list) ;runk_ = []
for mark in mark_set:
    runk_.append((mark_list.count(mark),mark))
runk_.sort(reverse=True)
print(runk_)
