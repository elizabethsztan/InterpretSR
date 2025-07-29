.. image:: _static/InterpretSR_header.png
   :alt: Description
   :width: 100%

*InterpretSR* allows you to approximate the behaviour of multi-layer perceptrons (MLPs) in deep learning models with symbolic equations using *`PySR <https://ai.damtp.cam.ac.uk/pysr/>`_*.


Key Features
============

* **Neural Network Wrapping**: Seamlessly wrap any PyTorch MLP with symbolic regression capabilities
* **Automatic Discovery**: Use *PySR* to find mathematical expressions that approximate neural network behavior
* **Interpretability**: Convert black-box neural networks into human-readable symbolic expressions
* **Flexible Integration**: Maintain full PyTorch compatibility while adding interpretability features
* **Dynamic Switching**: Switch between neural network and symbolic equation modes during inference

Installation
============

To install *InterpretSR*,

.. code-block:: bash

   pip install interpretsr

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   api_reference


