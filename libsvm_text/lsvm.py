from libsvm.python.svmutil import *
# from libsvm.python.svm import *
#
y, x = [1, -1], [{1:1, 2:1}, {1:-1, 2:-1}]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 4 -b 1')
# model = svm_train(prob, param,'-c 5')
model = svm_train(y, x,'-c 5')

yt = [1]
xt = [{1:1, 2:1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)



# from libsvm.python.svm import *
# from libsvm.python.svmutil import *
# y, x = [1, -1], [{1:1, 2:1}, {1:-1, 2:-1}]
# m = svm_train(y, x, '-c 5')


