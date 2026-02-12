# Two-Chunk Carver Reversal
This repository contains code for reversing a pair of carver seeds and the desired offset between the two chunks (on a single axis, either x or z) into a list of world seeds and coordinates at which such carver seeds can be found.

### What are carver seeds?
Carver seeds are special kinds of 48-bit Java Random seeds that Minecraft calculates for every single chunk in the game, based of the world seed and the chunk's coordinates. Carver seeds determine the layouts of most structures that may happen to spawn in a chunk, as well as the shapes of any caves or ravines spawning in the chunk.

The carver seed of chunk `(x, z)` for seed `world_seed` is calculated as follows:
```java
Random rand = new Random(s); // calls setSeed(s)
long a = rand.nextLong();
long b = rand.nextLong();
long carver_seed = (a * x) ^ (b * z) ^ world_seed;
```

### What exactly does this code do?
Let's say you want to find a seed and set of chunk coordinates (x,z) where two conditions are met at the same time:
- the chunk at (x, z) has a carver seed of your choice, for example `12345`
- the chunk at (x+N, z) or (x, z+N), where N is an integer constant of your choice, also has a carver seed of your choice, for example `54321`

The code lets you find triples of the form (seed, x, z) that satisfy both conditions at once.

### How does it work?
An explanation of the algorithm can be found in `cpp/two_carver_reversal.cpp`.
For a usage example, see `cpp/example.cpp`.
