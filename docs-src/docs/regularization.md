# Regularization

In their book [Deep Learning](#deep-learning) Ian Goodfellow et al. define regularization as
> "any modification we make to a learning algorithm that is intended to reduce its generalization error, but not its training error."

PyTorch's [optimizers](http://pytorch.org/docs/master/optim.html) use \\(l_2\\) parameter regularization to limit the capacity of models (i.e. reduce the variance).

In general, we can write this as:
\\[
loss(W;x;y) = loss_D(W;x;y) + \lambda_R R(W)
\\]
And specifically,
\\[
loss(W;x;y) = loss_D(W;x;y) + \lambda_R \lVert W \rVert_2^2
\\]
Where W is the collection of all weight elements in the network (i.e. this is model.parameters()), \\(loss(W;x;y)\\) is the total training loss, and \\(loss_D(W)\\) is the data loss (i.e. the error of the objective function, also called the loss function, or ```criterion``` in the Distiller sample image classifier compression application).
```
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
...
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```
\\(\lambda_R\\) is a scalar called the *regularization strength*, and it balances the data error and the regularization error.  In PyTorch, this is the ```weight_decay``` argument.

\\(\lVert W \rVert_2^2\\) is the square of the \\(l_2\\)-norm of W, and as such it is a *magnitude*, or sizing, of the weights tensor.
\\[
\lVert W \rVert_2^2 = \sum_{l=1}^{L}  \sum_{i=1}^{n} |w_{l,i}|^2 \;\;where \;n = torch.numel(w_l)
\\]

\\(L\\) is the number of layers in the network; and the notation about used 1-based numbering to simplify the notation.

The qualitative differences between the \\(l_2\\)-norm, and the squared \\(l_2\\)-norm is explained in [Deep Learning](https://www.deeplearningbook.org/).

## Sparsity and Regularization

We mention regularization because there is an interesting interaction between regularization and some DNN sparsity-inducing methods.

In [Dense-Sparse-Dense (DSD)](#han-et-al-2017), Song Han et al. use pruning as a regularizer to improve a model's accuracy:
> "Sparsity is a powerful form of regularization. Our intuition is that, once the network arrives at a local minimum given the sparsity constraint, relaxing the constraint gives the network more freedom to escape the saddle point and arrive at a higher-accuracy local minimum."

Regularization can also be used to induce sparsity.  To induce element-wise sparsity we can use the \\(l_1\\)-norm, \\(\lVert W \rVert_1\\).
\\[
\lVert W \rVert_1 = l_1(W) = \sum_{i=1}^{|W|} |w_i|
\\]

\\(l_2\\)-norm regularization reduces overfitting and improves a model's accuracy by shrinking large parameters, but it does not force these parameters to absolute zero.  \\(l_1\\)-norm regularization sets some of the parameter elements to zero, therefore limiting the model's capacity while making the model simpler.  This is sometimes referred to as *feature selection* and gives us another interpretation of pruning.

[One](https://github.com/IntelLabs/distiller/blob/master/jupyter/L1-regularization.ipynb) of Distiller's Jupyter notebooks explains how the \\(l_1\\)-norm regularizer induces sparsity, and how it interacts with \\(l_2\\)-norm regularization.


If we configure ```weight_decay``` to zero and use \\(l_1\\)-norm regularization, then we have:
\\[
loss(W;x;y) = loss_D(W;x;y) + \lambda_R \lVert W \rVert_1
\\]
If we use both regularizers, we have:
\\[
loss(W;x;y) = loss_D(W;x;y) + \lambda_{R_2} \lVert W \rVert_2^2  + \lambda_{R_1} \lVert W \rVert_1
\\]

Class ```distiller.L1Regularizer``` implements \\(l_1\\)-norm regularization, and of course, you can also schedule regularization.
```
l1_regularizer = distiller.s(model.parameters())
...
loss = criterion(output, target) + lambda * l1_regularizer()
```

## Group Regularization

In Group Regularization, we penalize entire groups of parameter elements, instead of individual elements.  Therefore, entire groups are either sparsified (i.e. all of the group elements have a value of zero) or not.  The group structures have to be pre-defined.

To the data loss, and the element-wise regularization (if any), we can add group-wise regularization penalty.  We represent all of the parameter groups in layer \\(l\\) as \\( W_l^{(G)} \\), and we add the penalty of all groups for all layers.  It gets a bit messy, but not overly complicated:
\\[
loss(W;x;y) = loss_D(W;x;y) + \lambda_R R(W) + \lambda_g \sum_{l=1}^{L} R_g(W_l^{(G)})
\\]

Let's denote all of the weight elements in group \\(g\\) as \\(w^{(g)}\\).

\\[
R_g(w^{(g)}) = \sum_{g=1}^{G} \lVert w^{(g)} \rVert_g = \sum_{g=1}^{G} \sum_{i=1}^{|w^{(g)}|} {(w_i^{(g)})}^2
\\]
where \\(w^{(g)} \in w^{(l)} \\) and \\( |w^{(g)}| \\) is the number of elements in \\( w^{(g)} \\).


\\( \lambda_g \sum_{l=1}^{L} R_g(W_l^{(G)}) \\) is called the Group Lasso regularizer.  Much as in \\(l_1\\)-norm regularization we sum the magnitudes of all tensor elements, in Group Lasso we sum the magnitudes of element structures (i.e. groups).  
<br>
Group Regularization is also called Block Regularization, Structured Regularization, or coarse-grained sparsity (remember that element-wise sparsity is sometimes referred to as fine-grained sparsity).  Group sparsity exhibits regularity (i.e. its shape is regular), and therefore
it can be beneficial to improve inference speed.

[Huizi-et-al-2017](#huizi-et-al-2017) provides an overview of some of the different groups: kernel, channel, filter, layers.  Fiber structures such as matrix columns and rows, as well as various shaped structures (block sparsity), and even [intra kernel strided sparsity](#anwar-et-al-2015) can also be used.

```distiller.GroupLassoRegularizer``` currently implements most of these groups, and you can easily add new groups.

## References
<div id="deep-learning"></div> **Ian Goodfellow and Yoshua Bengio and Aaron Courville**.
    [*Deep Learning*](https://www.deeplearningbook.org/),
     arXiv:1607.04381v2,
    2017.

<div id="han-et-al-2017"></div> **Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Enhao Gong, Shijian Tang, Erich Elsen, Peter Vajda, Manohar Paluri, John Tran, Bryan Catanzaro, William J. Dally**.
    [*DSD: Dense-Sparse-Dense Training for Deep Neural Networks*](https://arxiv.org/abs/1607.04381),
     arXiv:1607.04381v2,
    2017.

<div id="huizi-et-al-2017"></div> **Huizi Mao, Song Han, Jeff Pool, Wenshuo Li, Xingyu Liu, Yu Wang, William J. Dally**.
    [*Exploring the Regularity of Sparse Structure in Convolutional Neural Networks*](https://arxiv.org/abs/1705.08922),
    arXiv:1705.08922v3,
    2017.

<div id="anwar-et-al-2015"></div> **Sajid Anwar, Kyuyeon Hwang, and Wonyong Sung**.
    [*Structured pruning of deep convolutional neural networks*](https://arxiv.org/abs/1512.08571),
    arXiv:1512.08571,
    2015
