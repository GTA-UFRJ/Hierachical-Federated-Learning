# Hierachical-Federated-Learning






If you find this code useful in your research, please consider citing:

    @article{souza2022sbrc,
    author = {Lucas Airam C. de Souza, Gustavo F. Camilo, Matteo Sammarco, Miguel Elias M. Campista, and Luís Henrique M. K. Costa},
    journal = {Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos},
    title = {Aprendizado Federado com Agrupamento Hierárquico de Clientes para Aumento da Acurácia},
    year = {2022}
    }

    @article{souza2022ecml,
    author = {Lucas Airam C. de Souza, Gustavo F. Camilo, Matteo Sammarco, Marcin Detyniecki, Miguel Elias M. Campista, and Luís Henrique M. K. Costa},
    journal = {},
    title = {Federated Learning with Hierarchical Clustering of Clients},
    year = {2022}
    }




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


