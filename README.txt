Author: Umar Tariq
Version: December 2024
This codes relies on the numpy, random, heapq, and matplotlib python libraries. It was also
built and tested in python 3.10.

To run the N-Queens solver, open a command-line interface and execute the program using 
the following format (provided you have all the libraries):	 

python nQueens.py <board_size> <method> 

Replace <board_size> with the desired size of the chessboard (e.g., 8 for an 8x8 board) and
<method> with one of the implemented algorithms: backtracking, hill_climbing, or a_star. 
After running the program, the terminal will display the time taken to find a solution and 
an additional relevant statistic based on the selected method. For hill_climbing, it shows 
the number of iterations performed, while for a_star, it reports the number of nodes expanded. 
If a solution is found, a graphical representation of the board will appear, showing the queen 
placements. Exit out of the graph window to end the code execution.