---
title: sampling
type: doc
---

# sampling

#### `boltzmann_sampling(numbers, k=1, with_replacement=False)`

Return k numbers from a boltzmann-sampling over the supplied numbers

- **`numbers`** : `List or np.ndarray`

   	numbers to sample

- **`k`** : `int`

	_Default: 1_

   	How many numbers to sample. Choosing `k=None` will yield a single number

- **`with_replacement`** : `Boolean` `Default: False`

   	Allow replacement or not

**Returns:** `list`, `np.ndarray` or a single number (depending on the input)

__________________
 
#### `weighted_sampling(numbers, k=1, with_replacement=False)`

Return k numbers from a weighted-sampling over the supplied numbers

- **`numbers`** : `List or np.ndarray`

   	numbers to sample

- **`k`** : `int`
 
  	_Default: 1_
  
   	How many numbers to sample. Choosing `k=None` will yield a single number

- **`with_replacement`** : `Boolean` 

  	_Default: False_

   	Allow replacement or not

**Returns:** `list`, `np.ndarray` or a single number (depending on the input)