# QE Parallel Tuning Notes (Runtime-Focused)

Source: `/home/jackm/SOURCES/q-e-qe-7.4/Doc/user_guide.tex`

This file extracts only runtime-relevant guidance for parallel execution and tuning:
- Parallel execution hierarchy (images, pools, bands, PW, task groups, linear-algebra group)
- Runtime parallel flags (`-nimage`, `-npools`, `-nband`, `-ntg`, `-ndiag`, `-northo`)
- Practical execution/tuning notes and parallel I/O implications

Install/build details are intentionally omitted.

## Quick Runtime Summary

- Parallel hierarchy in QE is nested: `world -> images -> pools -> bands -> PW/tasks -> linear algebra group`.
- Primary runtime decomposition switches:
  - `-nimage` (`-ni`)
  - `-npools` (`-nk`)
  - `-nband` (`-nb`)
  - `-ntg` (`-nt`)
  - `-ndiag` (`-nd`) or `-northo`
- Default runtime values noted in QE docs:
  - `-ni 1 -nk 1 -nt 1`
  - `-nd` chosen from available per-pool processors (or 1 without ScaLAPACK)
- For large runs, QE explicitly recommends combining levels (`-nt`, `-nd`, plus MPI/OpenMP if configured).
- Parallel I/O mode and filesystem visibility can be limiting factors for restart portability and performance.

## Extracted Excerpts (with source line numbers)

### A) Parallel paradigms and execution setup (lines 1362-1423)

```text
 1362  \subsection{Understanding Parallelism}
 1363  
 1364  Two different parallelization paradigms are currently implemented
 1365  in \qe:
 1366  \begin{enumerate}
 1367  \item {\em Message-Passing (MPI)}. A copy of the executable runs
 1368  on each CPU; each copy lives in a different world, with its own
 1369  private set of data, and communicates with other executables only
 1370  via calls to MPI libraries. MPI parallelization requires compilation
 1371  for parallel execution, linking with MPI libraries, execution using
 1372  a launcher program (depending upon the specific machine). The number
 1373  of CPUs used
 1374  is specified at run-time either as an option to the launcher or
 1375  by the batch queue system.
 1376  \item {\em OpenMP}.  A single executable spawn subprocesses
 1377  (threads) that perform in parallel specific tasks.
 1378  OpenMP can be implemented via compiler directives ({\em explicit}
 1379  OpenMP) or via {\em multithreading} libraries  ({\em library} OpenMP).
 1380  Explicit OpenMP require compilation for OpenMP execution;
 1381  library OpenMP requires only linking to a multithreading
 1382  version of the mathematical libraries.
 1383  The number of threads is specified at run-time in the environment
 1384  variable OMP\_NUM\_THREADS.
 1385  \end{enumerate}
 1386  
 1387  MPI is the well-established, general-purpose parallelization.
 1388  In \qe\ several parallelization levels, specified at run-time
 1389  via command-line options to the executable, are implemented
 1390  with MPI. This is your first choice for execution on a parallel
 1391  machine.
 1392  
 1393  The support for explicit OpenMP is steadily improving.
 1394  Explicit OpenMP can be used together with MPI and also
 1395  together with library OpenMP. Beware
 1396  conflicts between the various kinds of parallelization!
 1397  If you don't know how to run MPI processes
 1398  and OpenMP threads in a controlled manner, forget about mixed
 1399  OpenMP-MPI parallelization.
 1400  
 1401  \subsection{Running on parallel machines}
 1402  
 1403  Parallel execution is strongly system- and installation-dependent.
 1404  Typically one has to specify:
 1405  \begin{enumerate}
 1406  \item a launcher program such as \texttt{mpirun} or \texttt{mpiexec},
 1407    with the  appropriate options (if any);
 1408  \item the number of processors, typically as an option to the launcher
 1409    program;
 1410  \item the program to be executed, with the proper path if needed;
 1411  \item other \qe-specific parallelization options, to be
 1412    read and interpreted by the running code.
 1413  \end{enumerate}
 1414  Items 1) and 2) are machine- and installation-dependent, and may be
 1415  different for interactive and batch execution. Note that large
 1416  parallel machines are  often configured so as to disallow interactive
 1417  execution: if in doubt, ask your system administrator.
 1418  Item 3) also depend on your specific configuration (shell, execution path, etc).
 1419  Item 4) is optional but it is very important
 1420  for good performances. We refer to the next
 1421  section for a description of the various
 1422  possibilities.
 1423  
```

### B) Parallel hierarchy and communication model (lines 1424-1500)

```text
 1424  \subsection{Parallelization levels}
 1425  
 1426  In \qe\ several MPI parallelization levels are
 1427  implemented, in which both calculations
 1428  and data structures are distributed across processors.
 1429  Processors are organized in a hierarchy of groups,
 1430  which are identified by different MPI communicators level.
 1431  The groups hierarchy is as follow:
 1432  \begin{itemize}
 1433  \item {\bf world}: is the group of all processors (MPI\_COMM\_WORLD).
 1434  \item
 1435  {\bf images}: Processors can then be divided into different "images", 
 1436  each corresponding to a different self-consistent or linear-response
 1437  calculation, loosely coupled to others. 
 1438  \item
 1439  {\bf pools}: each image can be subpartitioned into
 1440  "pools", each taking care of a group of k-points.
 1441  \item
 1442  {\bf bands}: each pool is subpartitioned into
 1443  "band groups", each taking care of a group
 1444  of Kohn-Sham orbitals (also called bands, or
 1445  wavefunctions). Especially useful for calculations
 1446  with hybrid functionals.
 1447  \item
 1448  {\bf PW}: orbitals in the PW basis set,
 1449  as well as charges and density in either
 1450  reciprocal or real space, are distributed
 1451  across processors.
 1452  This is usually referred to as "PW parallelization".
 1453  All linear-algebra operations on array of  PW /
 1454  real-space grids are automatically and effectively parallelized.
 1455  3D FFT is used to transform electronic wave functions from
 1456  reciprocal to real space and vice versa. The 3D FFT is
 1457  parallelized by distributing planes of the 3D grid in real
 1458  space to processors (in reciprocal space, it is columns of
 1459  G-vectors that are distributed to processors).
 1460  \item
 1461  {\bf tasks}:
 1462  In order to allow good parallelization of the 3D FFT when
 1463  the number of processors exceeds the number of FFT planes,
 1464  FFTs on Kohn-Sham states are redistributed to
 1465  ``task'' groups so that each group
 1466  can process several wavefunctions at the same time.
 1467  Alternatively, when this is not possible, a further
 1468  subdivision of FFT planes is performed.
 1469  \item
 1470  {\bf linear-algebra group}:
 1471  A further level of parallelization, independent on
 1472  PW or k-point parallelization, is the parallelization of
 1473  subspace diagonalization / iterative orthonormalization.
 1474   Both operations required the diagonalization of
 1475  arrays whose dimension is the number of Kohn-Sham states
 1476  (or a small multiple of it). All such arrays are distributed block-like
 1477  across the ``linear-algebra group'', a subgroup of the pool of processors,
 1478  organized in a square 2D grid. As a consequence the number of processors
 1479  in the linear-algebra group is given by $n^2$, where $n$ is an integer;
 1480  $n^2$ must be smaller than the number of processors in the PW group.
 1481  The diagonalization is then performed
 1482  in parallel using standard linear algebra operations.
 1483  (This diagonalization is used by, but should not be confused with,
 1484  the iterative Davidson algorithm). The preferred option is to use
 1485  ELPA and ScaLAPACK; alternative built-in algorithms are anyway available.
 1486  \end{itemize}
 1487  Note however that not all parallelization levels
 1488  are implemented in all codes.
 1489  
 1490  When a communicator is split, the MPI process IDs in each sub-communicator
 1491  remain ordered. So for instance, for two images and $2n$ MPI processes,
 1492  image 0 contains IDs $0,1,...,n-1$, image 1 contains IDs $n,n+1,..,2n-1$.
 1493  
 1494  \paragraph{About communications}
 1495  Images and pools are loosely coupled: inter-processors communication
 1496  between different images and pools is modest. Processors within each
 1497  pool are instead tightly coupled and communications are significant.
 1498  This means that fast communication hardware is needed if
 1499  your pool extends over more than a few processors on different nodes.
 1500  
```

### C) Runtime flags and tuning example (lines 1501-1537)

```text
 1501  \paragraph{Choosing parameters}:
 1502  To control the number of processors in each group,
 1503  command line switches:
 1504  \texttt{-nimage}, \texttt{-npools}, \texttt{-nband},
 1505  \texttt{-ntg}, \texttt{-ndiag} or \texttt{-northo}
 1506  (shorthands, respectively: \texttt{-ni}, \texttt{-nk}, \texttt{-nb},
 1507  \texttt{-nt}, \texttt{-nd})
 1508  are used.
 1509  As an example consider the following command line:
 1510  \begin{verbatim}
 1511  mpirun -np 4096 ./neb.x -ni 8 -nk 2 -nt 4 -nd 144 -i my.input
 1512  \end{verbatim}
 1513  This executes a NEB calculation on 4096 processors, 8 images (points in the configuration
 1514  space in this case) at the same time, each of
 1515  which is distributed across 512 processors.
 1516  k-points are distributed across 2 pools of 256 processors each,
 1517  3D FFT is performed using 4 task groups (64 processors each, so
 1518  the 3D real-space grid is cut into 64 slices), and the diagonalization
 1519  of the subspace Hamiltonian is distributed to a square grid of 144
 1520  processors (12x12).
 1521  
 1522  Default values are: \texttt{-ni 1 -nk 1 -nt 1} ;
 1523  \texttt{nd} is set to 1 if ScaLAPACK is not compiled,
 1524  it is set to the square integer smaller than or equal to the number of
 1525  processors of each pool.
 1526  
 1527  \paragraph{Massively parallel calculations}
 1528  For very large jobs (i.e. O(1000) atoms or more) or for very long jobs,
 1529  to be run on massively parallel  machines (e.g. IBM BlueGene) it is
 1530  crucial to use in an effective way all available parallelization levels:
 1531  on linear algebra (requires compilation with ELPA and/or ScaLAPACK),
 1532  on "task groups" (requires run-time option "-nt N"), and mixed
 1533  MPI-OpenMP (requires OpenMP compilation: \configure --enable-openmp).
 1534  Without a judicious choice of parameters, large jobs will find a
 1535  stumbling block in either memory or CPU requirements. Note that I/O
 1536  may also become a limiting factor.
 1537  
```

### D) Parallel I/O behavior that affects run strategy (lines 1538-1584)

```text
 1538  \subsection{Understanding parallel I/O}
 1539  In parallel execution, each processor has its own slice of data
 1540  (Kohn-Sham orbitals, charge density, etc), that have to be written
 1541  to temporary files during the calculation,
 1542  or to data files at the end of the calculation.
 1543  This can be done in two different ways:
 1544  \begin{itemize}
 1545  \item ``collected'': all slices are
 1546  collected by the code to a single processor
 1547  that writes them to disk, in a single file,
 1548  using a format that doesn't depend upon
 1549  the number of processors or their distribution.
 1550  This is the default since v.6.2 for final data.
 1551  \item ``portable'': as above, but data can be
 1552  copied to and read from a different machines
 1553  (this is not guaranteed with Fortran binary files).
 1554  Requires compilation with \verb|-D__HDF5|
 1555  preprocessing option and HDF5 libraries.
 1556  \end{itemize}
 1557  There is a third format, no longer used for final
 1558  data but used for scratch and restart files:
 1559  \begin{itemize}
 1560  \item ``distributed'': each processor
 1561  writes its own slice to disk in its internal
 1562  format to a different file.
 1563  The ``distributed'' format is fast and simple,
 1564  but the data so produced is readable only by
 1565  a job running on the same number of processors,
 1566  with the same type of parallelization, as the
 1567  job who wrote the data, and if all
 1568  files are on a file system that is visible to all
 1569  processors (i.e., you cannot use local scratch
 1570  directories: there is presently no way to ensure
 1571  that the distribution of processes across
 1572  processors will follow the same pattern
 1573  for different jobs).
 1574  \end{itemize}
 1575  
 1576  The directory for data is specified in input variables
 1577  \texttt{outdir} and \texttt{prefix} (the former can be specified
 1578  as well in environment variable ESPRESSO\_TMPDIR):
 1579  \texttt{outdir/prefix.save}. A copy of pseudopotential files
 1580  is also written there. If some processor cannot access the
 1581  data directory, the pseudopotential files are read instead
 1582  from the pseudopotential directory specified in input data.
 1583  Unpredictable results may follow if those files
 1584  are not the same as those in the data directory!
```

### E) Additional runtime caution from QE docs

QE also cautions against naive strong-scaling attempts when benchmarking parallelism (`user_guide.tex`, around line 964).

