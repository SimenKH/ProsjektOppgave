#pragma once
#include <vector>
#include <numeric>

class converter
{
public:
	double basic_circuit_and_loss();
	double switching_losses();
	double conduction_losses();

	double total_losses();

private:

};
