import torch
import numpy as np
import torch.nn as nn

def testShape():
    n = np.zeros(shape=(210,160,3))
    print(n)

# def testNN():
#     l = nn.Linear(2, 5)
#     v = torch.FloatTensor([[1, 2],[3,4]])
#     r = l(v)
#     print(r)
#
# def testPytorch():
#     a = torch.FloatTensor(3,4,5)
#     print(a)
#     a.zero_()
#     print(a.cuda())
#     print("device:",a.device())
#
#
#     b = torch.FloatTensor([[1, 2, 3], [3, 2, 1]])
#     print(b)
#
#     n = np.zeros(shape=(3, 2))
#     print(n)
#
#     d = torch.tensor(n)
#     print(d)
#
#     m = np.zeros(shape=(3, 2), dtype=np.float32)
#     e=torch.tensor(m)
#     print(e)
#
#     #torch.tensor接收一个numpy array作为参数；
#     l = torch.tensor([1, 2, 3])
#     print("l:",l)
#     print("sum of l is :",l.sum())
#
#     h = torch.tensor(99)
#     print("h is :",h)
#     print("h's item is:",h.item())
#
# def testGradient():
#     v1 = torch.tensor([4.0, 1.0], requires_grad=True);
#     v2 = torch.tensor([2.0, 2.0], requires_grad=True)
#     v_sum = v1 - v2
#     v_res = (v_sum * 4).sum()
#     print(v_res)
#     v_res.backward()
#     #v_res.backward()
#     #v_sum.backward()
#     #1、对叶子节点计算gradiant;2、只对requires_grad=true的节点进行计算
#     print(v1.grad)#gradient是一个向量，而导数是一个标量；
#     print(v2.grad)
#
# def testNN():
#     l = nn.Linear(2, 5)
#     v = torch.FloatTensor([[1, 2],[3,4]])
#     r = l(v)
#     print(r)
#
# def testLayer():
#     v = torch.FloatTensor([[1, 2], [3, 4]])
#     s = nn.Sequential(
#         nn.Linear(2, 5),
#         nn.ReLU(),
#         nn.Linear(5, 20),
#         nn.ReLU(),
#         nn.Linear(20, 10),
#         nn.Dropout(p=0.3),
#         nn.Softmax(dim=1)
#     )
#     r = s(v)
#     print(r)

if __name__ == "__main__":
    testShape()