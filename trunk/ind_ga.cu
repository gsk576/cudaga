#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "ind_ga.h"

__device__ int parseBits(unsigned char* bits, unsigned char* buffer);
__device__ void fillGeneRandom(unsigned char* bits, unsigned *seed);

__device__ int create_individual(chromo *parents, chromo *child, unsigned *seed)
{
	int xpoint = grand(seed) * GENE_BYTES;
	int i;
	if (!parents || !child)
		return -1;
	
	child->fitness = 0;
	child->result = 0;

	for (i = 0; i < xpoint; i++) {
		child->bits[i] = parents[0].bits[i];
	}

	for (; i < GENE_BYTES; i++) {
		child->bits[i] = parents[1].bits[i];
	}

	for (i = 0; i < (GENE_BYTES * 8); i++) {
		if ((grand(seed)) < MUTATION_RATE) {
			child->bits[i / 8] ^= (1 << (i % 8));
		}
	}

	return 0;
}

__device__ int init_individual(chromo *ind, unsigned *seed)
{
	if (!ind) return -1;
	
	ind->fitness = 0;
	fillGeneRandom((unsigned char *)ind->bits, seed);

	return 0;
}

__device__ void cpy_ind(chromo *ind, chromo *old)
{
	int i;
	ind->fitness = old->fitness;
	ind->result = old->result;
	for (i = 0; i < GENE_BYTES; i++) {
		ind->bits[i] = old->bits[i];
	}

	/*for (i = 0; i < (2 * GENE_BYTES); i++) {
		ind->buffer[i] = old->buffer[i];
	}*/

	return;
}

__device__ float gabs(float val) 
{
	if (val > 0) {
		return val;
	} else {
		return -val;
	}
}

__device__ int calc_fitness(chromo *ind)
{
	return calc_fitness_one(ind) + calc_fitness_one(ind + 1);
}

__device__ int calc_fitness_one(chromo *ind)
{
	float result = 0;
	int numElements;
	unsigned char buffer[2 * GENE_BYTES];
	int i;

	if (!ind) return -1;
	
	// Relegate the parsing of the bits to a specific function.
	// In this case, each byte of genes can hold two mathematical
	// things (operators or numbers).  Allocate for the worst case.
	
	// Now the buffer should have a correct series of number/op/number etc
	//numElements = parseBits((unsigned char *)ind->bits, ind->buffer);
	numElements = parseBits((unsigned char *)ind->bits, buffer);
    for (i = 0; (i < numElements - 1) && ((i + 1) < (2 * GENE_BYTES)); i += 2) {
		switch (buffer[i]) {
		  case 10:
			result += buffer[i+1];
			break;
		  case 11:
			result -= buffer[i+1];
			break;
		  case 12:
			result *= buffer[i+1];
			break;
		  case 13:
			result /= buffer[i+1];
			break;
		  default:
			break;
		}
	}

	// Now calculate the fitness
	// They use a very high fitness to say solution has been found.
	// Go with that for now
	ind->result = result;
	if (result == TARGET_VALUE)
		ind->fitness = 9999;
	else
		ind->fitness = ((int)1000.0/gabs(TARGET_VALUE - result));	

	return 0;
}


// ////////////////////
// Supporting Functions
// ////////////////////

// Parses the bits of an individual into a valid string of mathematical
// operators.  This is number, operator, number, etc.  Anything else is
// thrown out.  Fills the passed buffer with the ops and returns the
// number of elements
__device__ int parseBits(unsigned char* bits, unsigned char* buffer)
{
	unsigned char isOperator = 1;	// Want an operator next
	unsigned char temp;
	int index = 0;
	int i;

	for (i = 0; i < GENE_BYTES; i++) {
		temp = (bits[i] & 0x0F);
		if (isOperator) {
			if (!((temp < 10) || (temp > 13))) {
				isOperator = 0;
				buffer[index++] = temp;
			}
		} else {
			if (!(temp > 9)) {
				isOperator = 1;
				buffer[index++] = temp;
			}
		}
		temp = (bits[i] & 0xF0) >> 4;
		if (isOperator) {
			if (!((temp < 10) || (temp > 13))) {
				isOperator = 0;
				buffer[index++] = temp;
			}
		}
		else {
			if (!(temp > 9)) {
				isOperator = 1;
				buffer[index++] = temp;
			}
		}
	}
	
	// Now the original authors removed any divide by zero risks
	// by replacing any '/' that's followed by a '0' with a '+'

	for (i = 0; i < (index - 1); i++) {
		if ((buffer[i] == 13) && (buffer[i+1] == 0))
			buffer[i] = 10;
	}

	return index;
}

__device__ void fillGeneRandom(unsigned char* bits, unsigned *seed)
{
	int i;

	if (!bits) return;

	for (i = 0; i < GENE_BYTES; i++) {
		bits[i] = (int)(grand(seed) * 255.0);
	}
	return;
}

