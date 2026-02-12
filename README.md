# Two-Chunk Carver Reversal
This repository contains code for reversing a pair of carver seeds and the desired offset between the two chunks (on a single axis, either x or z) into a list of world seeds and coordinates at which such carver seeds can be found.

### What are carver seeds?
Carver seeds are special kinds of 48-bit Java Random seeds that Minecraft calculates for every single chunk in the game, based of the world seed and the chunk's coordinates. Carver seeds determine the layouts of most structures that may happen to spawn in a chunk, as well as the shapes of any caves or ravines spawning in the chunk.

The carver seed of chunk `(x, z)` for seed `world_seed` is calculated as follows:
```java
Random rand = new Random(s);  calls setSeed(s)
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

Let's assume the two carver chunks are at the same z coordinate. We have <br>
    `ax     ^ bz ^ seed = C1  (mod 2**48)  [C1 = carver seed of first chunk]`<br>
    `a(x+D) ^ bz ^ seed = C2  (mod 2**48)  [C2 = carver seed of second chunk]`

Xoring the equations:<br>
    `ax ^ a(x+D) = C1 ^ C2 (mod 2**48)`

We can bruteforce all 3.75 million possible values of x. 
For each such value we get an equation of the form<br>
    `Ma ^ Na = C1 ^ C2  (mod 2**48)`<br>
where a is the only unknown.
This can be solved iteratively, by reconstructing a from the lowest bits upwards (see `hensel_lift`).

In carver seeding, a is the value of a Java Random nextLong and can be reversed
back to the Java Random internal state using Matthew Bolan's NextLongReverser code.
That in turn gives us the structure seed, and all we're missing now is the z coordinate value.

To recover z, let's transform the first carver seed equation:<br>
    `ax ^ bz ^ seed = C1  (mod 2**48)`<br>
    `bz = ax ^ seed ^ C1  (mod 2**48)`<br>
Notice that everything on the right-hand-side is already known, let RHS = R.<br>
    `bz = R  (mod 2**48)`

If b is odd, we can directly calculate its inverse modulo 2**48 and multiply both sides by that.
Otherwise, we need to eliminate all the factors of 2 first, which is possible if and only if 
R has at least as many factors of 2 as b (if it's not possible the current x value yields no results). 
Let p be the number of excluded factors of 2. Then:<br>
    `(b >> p)z = (R >> p)  (mod 2**(48-p))`<br>
and we can calculate the mod inverse of b>>p instead, giving<br>
    `z = (R >> p) * modinv((b >> p), 2**(48-p))  (mod 2**(48-p))`

This z value is under the reduced modulo, and we're targetting modulo 2**48.
Therefore, we get 2**p valid solutions for z. Fortunately, since p is usually small, it's
sufficient to calculate the z value nearest to 0 under the reduced mod, as it will be the
only reasonable candidate under the original mod as well.

Finally, we have the value of z mod 2**(48-p), and we can map it back to the actual chunk z coordinate
by treating it as a signed U2 value stored on 48-p bits. That gives two valid ranges for z:
`[0, 1875000]` and `[2**(48-p) - 1875000, 2**(48-p))`. If the z value falls into any of these, we get a result.

The exact same process can be applied when starting from different z coordinates and the same x coordinate.
In that case, the algorithm recovers b first, and calculates x in the final step.
