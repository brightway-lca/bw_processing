Utility functions
=================

``bw_processing.utils``
-----------------------

This file also defines the ``COMMON_DTYPE`` used for matrix construction:

.. code-block:: python

    COMMON_DTYPE = [
        ("row_value", np.uint32),
        ("col_value", np.uint32),
        ("row_index", np.uint32),
        ("col_index", np.uint32),
        ("uncertainty_type", np.uint8),
        ("amount", np.float32),
        ("loc", np.float32),
        ("scale", np.float32),
        ("shape", np.float32),
        ("minimum", np.float32),
        ("maximum", np.float32),
        ("negative", np.bool),
        ("flip", np.bool),
    ]

Creating numpy structured arrays
********************************

.. autofunction:: bw_processing.utils.create_structured_array

Correctly formatting metadata
*****************************

.. autofunction:: bw_processing.utils.create_datapackage_metadata
