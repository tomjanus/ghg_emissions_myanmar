# Dam-Portfolio-Selection-Expansion-and-Compression-CPAIOR
This is the source code for the paper Efficiently Approximating High-dimensional Pareto Frontiers for Tree-structured Networks using Expansion and Compression, CPAIOR 2023. 

# Basin Input Files:
We have the dam and river information of the full MYANMAR basin and four its subbasins, i.e., West MYANMAR (WA), maranon, tapajos and ucayali. They are put in the folder Basin_Input_Files.

# Usage:
	1. Firstly, run `make MYANMAR_lp` to compile the C++ code.
	2. ./MYANMAR_lp -lp -epsilon [APPROXIMATION_FACTOR] -path [YOUR_INPUT_FILE_PATH] -criteria INum cri_1 cri_2 cri_{INum - 1} cri_{INum} -w ENum n1 w1 ... w_{n1}  n_{ENum} w1 ... w_{n_{ENum}} 
where INum refers to the number of criteria you implicitly consider and ENum refers to the number of criteria you explicitly consider minus one. Here we minus one because the first criteria has to be the energy.

We are sorry that the input format is a little bit confusing. Here is an example for you understanding. Assume we want to implicitly consider criteria c1, c2, c3. And we want to explicitly consider two criteria and compress c2 and c3 into one criterion with weights 0.5 and 0.5. Then the command should be:

	./MYANMAR_lp -lp -epsilon [APPROXIMATION_FACTOR] -path [YOUR_INPUT_FILE_PATH] -criteria 3 c1 c2 c3 -w 1 2 0.5 0.5

The criteria name are listed below:

energy, connectivity, sediment, biodiversity, ghg, dor
