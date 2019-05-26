str1 = '123gg321'
# #
# #
# str2 = str1[::-1] # 反转1
# # # str2 = ''.join(reversed(str1)) # 反转2
# #
# # def rev_string(a_string):# 反转3
# #     l = list(a_string)
# #     l.reverse()
# #     print(''.join(l))
# # rev_string(str1)
# #
# if str1==str2:
#     print('是回文')、
if str1==str1[::-1]:
    print('是回文')

a = set('abress')
b = set('address')

print(a)

print(a - b)  # a 和 b 的差集   a有b无
print(a | b)  # a 和 b 的并集   并集
print(a & b)  # a 和 b 的交集   交集
print(a ^ b)  # a 和 b 中不同时存在的元素 交集外的部分