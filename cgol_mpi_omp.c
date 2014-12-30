/*
 * Conway's Game of Life
 * Parallel MPI implementation with selective OpenMP usage.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>
#ifdef USE_OMP
# include <omp.h>
#endif

#define SUCCESS		0
#define FAILURE		1
#define NOMEM		2
#define IOERROR		3
#define INVARG		4

#define TAG 0

enum direction {
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UPLEFT,
	UPRIGHT,
	DOWNLEFT,
	DOWNRIGHT
};

/*
 * Calculate the source and destination ranks for each neighboring process.
 */
static void setup(MPI_Comm comm_cart, int myrank, int dimx, int dimy,
				int source_ranks[8], int dest_ranks[8])
{
	int mycoords[2], neighbor_coords[2];

	assert(myrank >= 0);
	assert(dimx > 0 && dimy > 0);
	MPI_Cart_coords(comm_cart, myrank, 2, mycoords);
	/* Up */
	MPI_Cart_shift(comm_cart, 1, 1, &source_ranks[UP], &dest_ranks[UP]);
	/* Down */
	MPI_Cart_shift(comm_cart, 1, -1, &source_ranks[DOWN], &dest_ranks[DOWN]);
	/* Left */
	MPI_Cart_shift(comm_cart, 0, -1, &source_ranks[LEFT], &dest_ranks[LEFT]);
	/* Right */
	MPI_Cart_shift(comm_cart, 0, 1, &source_ranks[RIGHT], &dest_ranks[RIGHT]);

	/* Diagonals in an MPI grid topology need separate handling. */
	/* Modulo operations are fixed to be positive. */
	/* Up and Left */
	neighbor_coords[1] = (mycoords[1] + 1) % dimx;				/* Up */
	neighbor_coords[0] = (mycoords[0] - 1 + dimy) % dimy;		/* Left */
	MPI_Cart_rank(comm_cart, neighbor_coords, &dest_ranks[UPLEFT]);
	source_ranks[DOWNRIGHT] = dest_ranks[UPLEFT];
	/* Up and Right */
	neighbor_coords[1] = (mycoords[1] + 1) % dimx;				/* Up */
	neighbor_coords[0] = (mycoords[0] + 1) % dimy;				/* Right */
	MPI_Cart_rank(comm_cart, neighbor_coords, &dest_ranks[UPRIGHT]);
	source_ranks[DOWNLEFT] = dest_ranks[UPRIGHT];
	/* Down and Left */
	neighbor_coords[1] = (mycoords[1] -1 + dimx) % dimx;		/* Down */
	neighbor_coords[0] = (mycoords[0] - 1 + dimy) % dimy;		/* Left */
	MPI_Cart_rank(comm_cart, neighbor_coords, &dest_ranks[DOWNLEFT]);
	source_ranks[UPRIGHT] = dest_ranks[DOWNLEFT];
	/* Down and Right */
	neighbor_coords[1] = (mycoords[1] -1 + dimx) % dimx;		/* Down */
	neighbor_coords[0] = (mycoords[0] + 1) % dimy;				/* Right */
	MPI_Cart_rank(comm_cart, neighbor_coords, &dest_ranks[DOWNRIGHT]);
	source_ranks[UPLEFT] = dest_ranks[DOWNRIGHT];
}

/*
 * Apply the rules for a specific cell of the grid.
 */
static void calc_next_gen(char **grid, char **nextgrid, int i, int j)
{
	/* Sum of up, middle and down neighbors. */
	const int sum =
		grid[i-1][j-1] + grid[i-1][j] + grid[i-1][j+1] +
		grid[i][j-1] + grid[i][j+1] +
		grid[i+1][j-1] + grid[i+1][j] + grid[i+1][j+1];

	assert(sum >= 0 && sum <= 8);
	if (grid[i][j])		/* Alive */
		nextgrid[i][j] = (sum < 2 || sum > 3) ? 0 : 1;
	else
		nextgrid[i][j] = (sum == 3) ? 1 : 0;
}

/******************************************************************************/

/*
 * Play the Game of Life up to the generation specified.
 */
void run_simulation(char **grid, char **nextgrid, int generations, int nrows,
					int ncolumns, int myrank, MPI_Comm comm_cart, int dims[2])
{
	int source_ranks[8], dest_ranks[8];
	MPI_Request send_reqs[8], recv_reqs[8];
	const int imax = nrows - 2, jmax = ncolumns - 2;

	/* Create a column type in order to send the left and right columns
	 * of the grid. */
	MPI_Datatype coltype;
	MPI_Type_vector(imax, 1, ncolumns, MPI_CHAR, &coltype);
	MPI_Type_commit(&coltype);

	setup(comm_cart, myrank, dims[0], dims[1], source_ranks, dest_ranks);
	while (generations--) {
		int i, j;
		char **temp;
		MPI_Status status;

		MPI_Isend(&grid[1][1], jmax, MPI_CHAR, dest_ranks[UP],
			TAG, comm_cart, &send_reqs[UP]);
		MPI_Isend(&grid[imax][1], jmax, MPI_CHAR, dest_ranks[DOWN],
			TAG, comm_cart, &send_reqs[DOWN]);
		MPI_Isend(&grid[1][1], 1, coltype, dest_ranks[LEFT],
			TAG, comm_cart, &send_reqs[LEFT]);
		MPI_Isend(&grid[1][jmax], 1, coltype, dest_ranks[RIGHT],
			TAG, comm_cart, &send_reqs[RIGHT]);

		MPI_Isend(&grid[1][1], 1, MPI_CHAR, dest_ranks[UPLEFT],
			TAG, comm_cart, &send_reqs[UPLEFT]);
		MPI_Isend(&grid[1][jmax], 1, MPI_CHAR, dest_ranks[UPRIGHT],
			TAG, comm_cart, &send_reqs[UPRIGHT]);
		MPI_Isend(&grid[imax][1], 1, MPI_CHAR, dest_ranks[DOWNLEFT],
			TAG, comm_cart, &send_reqs[DOWNLEFT]);
		MPI_Isend(&grid[imax][jmax], 1, MPI_CHAR, dest_ranks[DOWNRIGHT],
			TAG, comm_cart, &send_reqs[DOWNRIGHT]);

		/* Receive the symmetric elements first. */
		MPI_Irecv(&grid[nrows-1][1], jmax, MPI_CHAR, source_ranks[DOWN],
			MPI_ANY_TAG, comm_cart, &recv_reqs[DOWN]);
		MPI_Irecv(&grid[0][1], jmax, MPI_CHAR, source_ranks[UP],
			MPI_ANY_TAG, comm_cart, &recv_reqs[UP]);
		MPI_Irecv(&grid[1][ncolumns-1], 1, coltype, source_ranks[RIGHT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[RIGHT]);
		MPI_Irecv(&grid[1][0], 1, coltype, source_ranks[LEFT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[LEFT]);

		MPI_Irecv(&grid[nrows-1][ncolumns-1], 1, MPI_CHAR, source_ranks[DOWNRIGHT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[DOWNRIGHT]);
		MPI_Irecv(&grid[nrows-1][0], 1, MPI_CHAR, source_ranks[DOWNLEFT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[DOWNLEFT]);
		MPI_Irecv(&grid[0][ncolumns-1], 1, MPI_CHAR, source_ranks[UPRIGHT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[UPRIGHT]);
		MPI_Irecv(&grid[0][0], 1, MPI_CHAR, source_ranks[UPLEFT],
			MPI_ANY_TAG, comm_cart, &recv_reqs[UPLEFT]);

		/* Compute the internal cells. */
#ifdef USE_OMP
# pragma omp parallel for schedule(static) num_threads(4)
#endif
		for (i = 2; i < imax; ++i)
			for (j = 2; j < jmax; ++j)
				calc_next_gen(grid, nextgrid, i, j);

		MPI_Wait(&recv_reqs[DOWN], &status);
		for (j = 2; j < jmax; ++j)
			calc_next_gen(grid, nextgrid, imax, j);
		MPI_Wait(&recv_reqs[UP], &status);
		for (j = 2; j < jmax; ++j)
			calc_next_gen(grid, nextgrid, 1, j);
		MPI_Wait(&recv_reqs[RIGHT], &status);
		for (i = 2; i < imax; ++i)
			calc_next_gen(grid, nextgrid, i, jmax);
		MPI_Wait(&recv_reqs[LEFT], &status);
		for (i = 2; i < imax; ++i)
			calc_next_gen(grid, nextgrid, i, 1);

		MPI_Wait(&recv_reqs[DOWNRIGHT], &status);
		calc_next_gen(grid, nextgrid, imax, jmax);
		MPI_Wait(&recv_reqs[DOWNLEFT], &status);
		calc_next_gen(grid, nextgrid, imax, 1);
		MPI_Wait(&recv_reqs[UPRIGHT], &status);
		calc_next_gen(grid, nextgrid, 1, jmax);
		MPI_Wait(&recv_reqs[UPLEFT], &status);
		calc_next_gen(grid, nextgrid, 1, 1);

		for (i = 0; i < 8; ++i)
			MPI_Wait(&send_reqs[i], &status);
		/* Swap the grids for the next generation. */
		temp = grid;
		grid = nextgrid;
		nextgrid = temp;
	}
	MPI_Type_free(&coltype);
}

/******************************************************************************/

/*
 * Allocate memory for a new grid of the specified size.
 * A two dimensional (but contiguous in memory) array is allocated
 * and returned.
 */
char **allocate_grid(size_t nrows, size_t ncolumns)
{
	char **grid;
	size_t i;

	if (nrows < 9 || ncolumns < 9)
		return NULL;
	/* Keep the array's contents contiguous. */
	if ((grid = malloc(nrows * sizeof (*grid))) == NULL)
		return NULL;
	if ((grid[0] = malloc(nrows * ncolumns * sizeof (**grid))) == NULL) {
		free(grid);
		return NULL;
	}
	for (i = 1; i < nrows; ++i)
		grid[i] = &(grid[0][i * ncolumns]);
	return grid;
}

/*
 * Free the memory of the grid specified.
 * The grid specified must be one created by 'allocated_grid'.
 */
void deallocate_grid(char **grid)
{
	assert(grid != NULL);
	free(grid[0]);
	free(grid);
}

/*
 * Generate random cells for the grid specified.
 * Only the inner cells are taken into account.
 */
void generate_grid(char **grid, size_t nrows, size_t ncolumns, int myrank)
{
	size_t i, j, imax = nrows - 1, jmax = ncolumns - 1;

	assert(grid != NULL);
	srand((unsigned int) (time(NULL) * myrank));
	for (i = 1; i < imax; ++i)
		for (j = 1; j < jmax; ++j)
			grid[i][j] = (char) rand() % 2;
}

/******************************************************************************/

/*
 * Just print out how to call the program.
 */
void usage(const char *progname)
{
	fprintf(stderr, "Usage: ./%s <generations> <N> <M> [<filename>]\n"
		"If no file name is given, a random grid is generated.\n", progname);
}

static int check_and_parse(int argc, char *argv[], int *generations,
				int *N, int *M, FILE **file)
{
	char *endptr;

	if (argc != 4 && argc != 5) {
		usage(argv[0]);
		return INVARG;
	}
	*generations = strtol(argv[1], &endptr, 10);
	if (*endptr != '\0' || *generations < 1) {
		fprintf(stderr, "<generations> must be a positive number.\n");
		return INVARG;
	}
	*N = strtol(argv[2], &endptr, 10);
	if (*endptr != '\0' || *N < 9) {
		fprintf(stderr, "N must be a number greater than 8.\n");
		return INVARG;
	}
	*M = strtol(argv[3], &endptr, 10);
	if (*endptr != '\0' || *M < 9) {
		fprintf(stderr, "M must be a number greater than 8.\n");
		return INVARG;
	}
	if (argc == 5)
		if ((*file = fopen(argv[4], "rb")) == NULL) {
			fprintf(stderr, "Error opening file \"%s\".\n", argv[4]);
			return IOERROR;
		}
	return SUCCESS;
}

/*
 * Parse the arguments of the program.
 * Only the process with rank zero checks the validity of the arguments.
 * In case of error, all the other processes are notified via an
 * MPI_Bcast call and then terminate normally.
 */
void parse_arguments(int argc, char *argv[], int myrank,
					int *generations, int *N, int *M, FILE **file)
{
	int error = SUCCESS;

	/* Only process with rank 0 is responsible for checking the arguments. */
	if (myrank == 0)
		error = check_and_parse(argc, argv, generations, N, M, file);
	/* Report any errors to other processes. */
	MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (error != SUCCESS) {
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}
	/* It's time for the rest of the processes to compute their arguments. */
	/* No need for error checking here (assume that the file is still ok). */
	if (myrank != 0) {
		*generations = strtol(argv[1], NULL, 10);
		*N = strtol(argv[2], NULL, 10);
		*M = strtol(argv[3], NULL, 10);
		if (argc == 5)
			*file = fopen(argv[4], "rb");
	}
}

int readfile(FILE *file, char **grid, size_t nrows, size_t ncolumns, int rank)
{
	size_t imax = nrows - 1, jmax = ncolumns - 1;
	size_t i, j, skip;

	skip = imax * jmax * (size_t) rank;
	for (i = 1; i < imax; ++i)
		for (j = 1; j < jmax; ++j)
			grid[i][j] = (char) getc(file);

	if (fclose(file) == EOF)
		return IOERROR;
	return SUCCESS;
}

int main(int argc, char *argv[])
{
	int generations, N, M;
	FILE *file;
	double start_time, end_time;
	char **grid, **nextgrid;
	int numprocs, myrank;
	int dims[2];
	int periods[2] = {1, 1};
	size_t dimx, dimy;
	MPI_Comm comm_cart;

	MPI_Init(&argc, &argv);
	/* Find out how big the SPMD world is. */
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	/* Find out what the process' rank is. */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	parse_arguments(argc, argv, myrank, &generations, &N, &M, &file);

	/* Assume that the number of processes divides the grid size evenly. */
	numprocs = (int) sqrt(numprocs);
	dims[0] = numprocs;
	dims[1] = numprocs;
	if (myrank == 0)
		printf("Creating 2D cartesian topology with %d processes in dimension "
				"X and %d in dimension Y.\n", dims[0], dims[1]);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);

	dimx = (size_t) (N / numprocs);
	dimy = (size_t) (M / numprocs);
	assert(dimx >= 9 && dimy >= 9);
	/* We need extra space for boundaries */
	dimx += 2;
	dimy += 2;
	if (myrank == 0)
		printf("Allocating grids with dimensions [%d, %d]...\n", dimx, dimy);
	if ((grid = allocate_grid(dimx, dimy)) == NULL)
		MPI_Abort(comm_cart, NOMEM);
	/* We need an extra grid for computing the new generation. */
	if ((nextgrid = allocate_grid(dimx, dimy)) == NULL) {
		deallocate_grid(grid);
		MPI_Abort(comm_cart, NOMEM);
	}

	/* Read the corresponding part of the file, if a file is given */
	if (argc == 5) {
		if (myrank == 0)
			puts("Reading file...");
		if (readfile(file, grid, dimx, dimy, myrank) == IOERROR) {
			deallocate_grid(grid);
			deallocate_grid(nextgrid);
			MPI_Abort(comm_cart, IOERROR);
		}
	}
	/* or generate a random grid. */
	else {
		if (myrank == 0)
			puts("Generating random grid...");
		generate_grid(grid, dimx, dimy, myrank);
	}

	if (myrank == 0)
		puts("Starting simulation...");
	/* Wait for all the processes before measuring time, for more accuracy. */
	MPI_Barrier(comm_cart);
	start_time = MPI_Wtime();
	run_simulation(grid, nextgrid, generations, (int) dimx, (int) dimy,
		myrank, comm_cart, dims);
	MPI_Barrier(comm_cart);
	end_time = MPI_Wtime();
	if (myrank == 0)
		printf("Wall clock time = %f\n", end_time - start_time);

	if (myrank == 0)
		puts("Cleaning up...");
	deallocate_grid(grid);
	deallocate_grid(nextgrid);
	if (myrank == 0)
		puts("Finalizing MPI...");
	MPI_Finalize();
	if (myrank == 0)
		puts("Exiting...");
	return EXIT_SUCCESS;
}