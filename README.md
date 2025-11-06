# PV Functions

Prof. Dr. Lucas Vizzotto Bellinaso 
Grupo de Eletrônica de Potência e Controle (GEPOC)
Instituto de Energia e Mobilidade (IEM)
Universidade Federal de Santa Maria (UFSM)
06/11/2025

How to run in Google Colab:

```python
!wget https://raw.githubusercontent.com/lucasbellinaso/PV_functions/main/functions.py
from functions import *

# To check available functions:
import inspect, functions
funcoes = [name for name, obj in inspect.getmembers(functions, inspect.isfunction)]
print("\nFunções disponíveis:\n", "\n".join(funcoes))
```
