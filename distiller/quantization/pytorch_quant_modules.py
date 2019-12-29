import torch.nn as nn
import torch.nn.quantized as nnq


class QFunctionalWrapper(nn.Module):
    def __init__(self):
        super(QFunctionalWrapper, self).__init__()
        self.qfunc = nnq.QFunctional()


class QFunctionalAdd(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAdd, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add(x, y)


class QFunctionalAddScalar(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAddScalar, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add_scalar(x, y)


class QFunctionalMul(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalMul, self).__init__()

    def forward(self, x, y):
        return self.qfunc.mul(x, y)


class QFunctionalMulScalar(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalMulScalar, self).__init__()

    def forward(self, x, y):
        return self.qfunc.mul_scalar(x, y)


class QFunctionalCat(QFunctionalWrapper):
    def __init__(self, dim=0):
        super(QFunctionalCat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return self.qfunc.cat(x, self.dim)


class QFunctionalAddRelu(QFunctionalWrapper):
    def __init__(self):
        super(QFunctionalAddRelu, self).__init__()

    def forward(self, x, y):
        return self.qfunc.add_relu(x, y)


class ConditionalDeQuantize(nn.Module):
    def __init__(self):
        super(ConditionalDeQuantize, self).__init__()

    def forward(self, x):
        if x.is_quantized:
            return x.dequantize()
        return x


class ConditionalQuantize(nn.Module):
    def __init__(self):
        super(ConditionalQuantize, self).__init__()

    def forward(self, x):
        if x.is_quantized:
            return x.dequantize()
        return x
