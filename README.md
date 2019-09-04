# dl-pytorch-selftrain

> Those notebooks are explorations I am making to train myself to the pytorch framework. As the best way to learn is to teach, I'm writing those as notebooks for others to train as well.

### Target Audience

This will likely work for you if you already **have a background and practice of Machine Learning** (with scikit-learn), and some **foundational knowledge on Deep Learning**.

Sometimes, because we know something well already (like... backprop, neural nets, scikit) we tend to skip some of the usual tutorials, and looking a few steps ahead we're stuck and confused because we didn't take time (or don't have it) to retrain from scratch. I'm trying to provide a short learning path for that situation.

### Requirements

**This is not a training to Machine Learning**.

To benefit from this content, it is preferable to know the following algorithms or concepts:
- Logistic Regression (log likelyhood, etc)
- Gradient Descent (gradient, learning rate, etc)
- Neural Networks (the 90's basic kind)

### Learning Objectives  

- discover how pytorch's autograd supports gradient computation for a given computation graph
- use autograd to compute gradient for a known modeling technique (LogReg)
- use pytorch `Module` to package a simple neural net on a well known ML problem (IRIS)

### Getting started

Using anaconda
```sh
conda env create --file env.yml
```

### Learning Path

The notebooks are using pytorch to decompose/recompose some usual the machine learning algorithms. We'll first rebuild them from scratch using low-level compoments (`autograd`)  of pytorth, to understand the foundations using known computation steps. Then we'll use high level calls (`Module`, etc) to do more fancy stuff.

Note: we're following along some of the content from Stanford [Deep Learning for Natural Language Processing - CS224n](http://web.stanford.edu/class/cs224n/index.html#schedule).

| Steps |
| :-- |
| [`notebooks/step-01-autograd101`](notebooks/step-01-autograd101.ipynb) - description TBD |
| [`notebooks/step-02-logreg-with-autograd`](notebooks/step-02-logreg-with-autograd.ipynb) - description TBD |
| [`notebooks/step-03-iris-torch-nn`](notebooks/step-03-iris-torch-nn.ipynb) - description TBD |
