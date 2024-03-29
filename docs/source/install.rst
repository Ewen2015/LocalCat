Installation Guide
******************

The General Approach
====================

LocalCat can be installed from `PyPI <https://pypi.org/project/localcat/>`_ with :code:`pip`.

.. code-block:: bash
  
  pip install localcat


The Latest Version
==================

For the lateset version, install from `GitHub <https://github.com/Ewen2015/LocalCat>`_:

.. code-block:: bash
  
  pip install git+https://github.com/Ewen2015/LocalCat


For Users in China
==================

This can be slow if you are in China. Try add :code:`--index-url`:

.. code-block:: bash
  
  pip install localcat -i https://pypi.douban.com/simple
  

Base URL of the Python Package Index (default https://pypi.org/simple). This should point to a repository compliant with PEP 503 (the simple repository API) or a local directory laid out in the same format.
  
.. code-block:: bash

  -i, --index-url <url>


.. list-table:: **PIP China Source List**
   :widths: 25 45
   :header-rows: 1

   * - Name
     - Address
   * - Douban
     - https://pypi.douban.com/simple
   * - Tsinghua University
     - https://pypi.tuna.tsinghua.edu.cn/simple
   * - University of Science and Technology of China
     - https://pypi.mirrors.ustc.edu.cn/simple
   * - Alibaba Cloud
     - https://mirrors.aliyun.com/pypi/simple
   * - Huawei Cloud
     - https://repo.huaweicloud.com/repository/pypi/simple
