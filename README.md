# Hierachical-Federated-Learning





## FAQ

### _lzma Import Failed

Solution: 

Install backports.lzma with pip
     
    python3.9 -m pip install backports.lzma

Open the file lzma.py:
      
    vim /usr/local/lib/python3.9/lzma.py 

Modify the line 27: 
               
     try:
         from _lzma import *
         from _lzma import _encode_filter_properties, _decode_filter_properties
     except ImportError:
         from backports.lzma import *
         from backports.lzma import _encode_filter_properties, _decode_filter_properties


