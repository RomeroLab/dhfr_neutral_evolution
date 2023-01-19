### Contact map for DHFR

[Numpy integer matrix](./contact_map.npy) of size (186, 186). 186 is the length of the DHFR protein.
  - An value of 5 marks pairs where the heavy atom distance is <5A.
  - An value of 8 marks pairs where the heavy atom distance is between 5A and 8A. 
  - An value of 0 marks pairs where the heavy atom distance is >8A.

```python
import numpy as np
mat = np.load("./contact_map.npy")
print(mat.shape) # (186, 186)
```
