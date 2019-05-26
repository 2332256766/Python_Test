# for x in range(1,9):
#     print(('*'+'.'*(x-1))*x)

# str = [('*'+'.'*(x-1))*x for x in range(1,9)]
# print('\n'.join(str))

print('\n'.join([('*'+'.'*(x-1))*x for x in range(1,9)]))
